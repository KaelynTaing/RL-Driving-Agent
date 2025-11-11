import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import random
import cv2

class CarlaEnv(gym.Env):
    """Gym environment with CARLA vehicle and front camera observation."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, host="host.docker.internal", port=2000, town="Town01"):
        super().__init__()

        # Connect to CARLA
        print(f"[CarlaEnv] Connecting to CARLA at {host}:{port}...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        print(f"[CarlaEnv] Connected! Map: {self.world.get_map().name}")

        # Vehicle and sensor placeholders
        self.vehicle = None
        self.camera = None
        self.camera_image = None
        self.sensor_list = []

        # Action: throttle, brake, steer
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),  # steer, throttle, brake
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Observation: 84x84 RGB image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Destroy old actors
        self._cleanup_actors()

        # Spawn vehicle at random spawn point
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Attach front camera
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "84")
        camera_bp.set_attribute("image_size_y", "84")
        camera_bp.set_attribute("fov", "110")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.sensor_list.append(self.camera)

        # Register callback to get camera image
        self.camera.listen(lambda image: self._process_image(image))

        # Wait a moment to receive first image
        self.world.tick()
        while self.camera_image is None:
            self.world.tick()

        return self.camera_image, {}

    def step(self, action):
        # Apply vehicle control
        steer, throttle, brake = float(action[0]), float(action[1]), float(action[2])
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.vehicle.apply_control(control)

        # Advance simulation
        self.world.tick()

        # Reward: placeholder (distance moved)
        reward = 0.0
        done = False
        truncated = False

        return self.camera_image, reward, done, truncated, {}

    def _process_image(self, image):
        # Convert CARLA raw image to numpy array (RGB)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        self.camera_image = array

    def _cleanup_actors(self):
        # Destroy old vehicle and sensors
        if self.camera is not None:
            self.camera.stop()
        for actor in self.sensor_list:
            actor.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()
        self.sensor_list = []
        self.vehicle = None
        self.camera = None

    def close(self):
        self._cleanup_actors()
        super().close()
