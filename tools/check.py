import json

# with open ("/share/tengjianing/songyuhao/segmentation/datasets/0609/train_118000.json") as f:
#     datas = json.load(f)
# with open ("/share/tengjianing/songyuhao/segmentation/datasets/0612/train_98000_p.json", "w") as output:
    
#     for data in datas:
#         if  not "icu10_large_model" in data["img_path"]:
#             output.write(json.dumps(data) + "\n")
            
import jsonlines


            
with jsonlines.open ("/share/tengjianing/songyuhao/segmentation/datasets/0612/train_98000_p.json") as f:
    datas = [_ for _ in f]
with open ("/share/tengjianing/songyuhao/segmentation/datasets/0617/train_68000_p.json", "w") as output:
    
    for data in datas:
        if  not "mapillary" in data["img_path"]:
            output.write(json.dumps(data) + "\n")