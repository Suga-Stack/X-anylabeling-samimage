import os

filtered_datas = []
# def get_all_val():
#     base = 'data/luster_data'
#     datas = []
#     for f in os.listdir(base):
#         for d in os.listdir(f'{base}/{f}'):
#             im_dir = f'{base}/{f}/{d}/val/JPEGImages'
#             gt_dir = f'{base}/{f}/{d}/val/annotations'
#             name = f'{f}_{d}'
#             if name in filtered_datas:
#                 continue
#             name = f'{base}/{f}/{d}'
#             assert(os.path.exists(im_dir))
#             assert(os.path.exists(gt_dir))
#             datas.append({
#                 "name": name,
#                 "im_dir": im_dir,
#                 "gt_dir": gt_dir,
#                 "im_ext": ".jpg",
#                 "gt_ext": ".png"
#             })
#     assert(len(datas) == 24)
#     datas.sort(key=lambda x: x['name'])
#     return datas
    
# def get_all_luster_train():
#     base = 'data/luster_data'
#     datas = []
#     for f in os.listdir(base):
#         for d in os.listdir(f'{base}/{f}'):
#             im_dir = f'{base}/{f}/{d}/train/JPEGImages'
#             gt_dir = f'{base}/{f}/{d}/train/annotations'
#             name = f'{f}_{d}'
#             if name in filtered_datas:
#                 continue
#             name = f'{base}/{f}/{d}'
#             if not os.path.exists(im_dir):
#                 # print(f"[Train]: {im_dir} does not exist.")
#                 continue
#             datas.append({
#                 "name": name,
#                 "im_dir": im_dir,
#                 "gt_dir": gt_dir,
#                 "im_ext": ".jpg",
#                 "gt_ext": ".png"
#             })
#     assert(len(datas) == 18)
#     datas.sort(key=lambda x: x['name'])
#     return datas



# def get_all_luster_train(percent):
#     base = 'data_final_3'
#     datas = []
#     for f in os.listdir(base):
#         im_dir = f'{base}/{f}/train_{percent}/img'
#         gt_dir = f'{base}/{f}/train_{percent}/mask'
#         name = f'{base}/{f}_{percent}'
#         datas.append({
#             "name": name,
#             "im_dir": im_dir,
#             "gt_dir": gt_dir,
#             "im_ext": ".jpg",
#             "gt_ext": ".png"
#         })
#     assert(len(datas) == 5)
#     datas.sort(key=lambda x: x['name'])
#     return datas


# def get_all_luster_val(percent):
#     base = 'data_final_3'
#     datas = []
#     for f in os.listdir(base):
#         im_dir = f'{base}/{f}/val_{percent}/img'
#         gt_dir = f'{base}/{f}/val_{percent}/mask'
#         print("!@!!!!!!!!!!!!!", im_dir, gt_dir)
#         name = f'{base}/{f}_{percent}'
#         datas.append({
#             "name": name,
#             "im_dir": im_dir,
#             "gt_dir": gt_dir,
#             "im_ext": ".jpg",
#             "gt_ext": ".png"
#         })
#     assert(len(datas) == 5)
#     datas.sort(key=lambda x: x['name'])
#     return datas




def get_all_luster_train(num):
    base = f'data_final_90_{num}'
    datas = []
    for f in os.listdir(base):
        im_dir = f'{base}/{f}/train/img'
        gt_dir = f'{base}/{f}/train/mask'
        print("!@!!!!!!!!!!!!!", im_dir, gt_dir)
        name = f'{base}/{f}'
        datas.append({
            "name": name,
            "im_dir": im_dir,
            "gt_dir": gt_dir,
            "im_ext": ".jpg",
            "gt_ext": ".png"
        })
    assert(len(datas) == 5)
    datas.sort(key=lambda x: x['name'])
    return datas


def get_all_luster_val(num):
    base = f'data_final_90_{num}'
    datas = []
    for f in os.listdir(base):
        im_dir = f'{base}/{f}/test/img'
        gt_dir = f'{base}/{f}/test/mask'
        print("!@!!!!!!!!!!!!!", im_dir, gt_dir)
        name = f'{base}/{f}'
        datas.append({
            "name": name,
            "im_dir": im_dir,
            "gt_dir": gt_dir,
            "im_ext": ".jpg",
            "gt_ext": ".png"
        })
    assert(len(datas) == 5)
    datas.sort(key=lambda x: x['name'])
    return datas