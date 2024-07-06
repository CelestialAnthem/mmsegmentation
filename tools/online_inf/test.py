import json, uuid
from oss_util import OssUpload


def upload_img(input_path):
    img_oss_path = ""
    png_name = input_path.split("/")[-1]
    img_oss_path = gen_path(1, png_name)

    # 上传image
    # os = OssUpload()
    ins = OssUpload.relative_path(img_oss_path)
    is_ok = ins.put_object_from_file(input_path)

    return img_oss_path


def gen_path(clip_id, video_name):
    print("video_name is %s" % video_name)
    prefix = "tos://haomo-public/lucas-generation/online_services/segementation"  #拼接tos桶路径  自己定  应该是放到public
    guid = str(uuid.uuid4())
    return f"{prefix}/{clip_id}/{guid}/{video_name}"


if __name__ == "__main__":
    upload_img()
    # data_path = "tos://lucas-data/release/preparation/card_data/icu30/data_label_postprocess/hds_20240120114505_7v84lafs95/hds_20240116054157_3buggxthbx/1691353670300449.json"
    # path  = data_path.split("/")
    # print(path)