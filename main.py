from lib import models

# from lib.dataset import coco


def main():
    model = models.get_fully_pretrained_pose_net()
    model.eval()
    print(model)
    # coco()


if __name__ == "__main__":
    main()
