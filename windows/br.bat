docker build -t carla_img .
docker run --name carla_cont -it -v .:/workspace carla_img