import os

def get_all_hx_train(num, obj):
    base = f'data_25_{num}'
    datas = []
    for f in obj:
        im_dir = f'{base}/{f}/train_25/img'
        gt_dir = f'{base}/{f}/train_25/mask'
        print("!@!!!!!!!!!!!!!", im_dir, gt_dir)
        name = f'{base}/{f}'
        datas.append({
            "name": name,
            "im_dir": im_dir,
            "gt_dir": gt_dir,
            "im_ext": ".jpg",
            "gt_ext": ".png"
        })
    assert(len(datas) == 1)
    return datas


def get_all_hx_val(num, obj):
    base = f'data_25_{num}'
    datas = []
    for f in obj:
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
    assert(len(datas) == 1)
    return datas