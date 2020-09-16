import json
import os
from fid.fid_score import fid_score

''''''

if __name__=='__main__':

    target_path=r'./202008fid'
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    #GT
    real_paths = r"/data1/liangshuaizhe/map_translate/202008_SH_toeval/gt_split"
    fake_paths = r"/data1/liangshuaizhe/map_translate/202008_SH_toeval/gt_split"

    layers = ['15', '16', '17', '18']
    for layer in layers:
        for numresult in range(5):
            real_path=os.path.join(real_paths,str(numresult),layer,'a')
            fake_path=os.path.join(fake_paths,str(numresult),layer,'b')
            fid = fid_score(real_path=real_path, fake_path=fake_path, gpu='1')
            print(f'===> fid score:{fid:.4f}')
            target=os.path.join(target_path,'GT_'+layer+'_'+str(numresult)+'fid.json')
            with open(target, 'a') as f:
                json.dump(fid, f)

    # noise
    real_paths=r"/data1/liangshuaizhe/map_translate/202008_SH_toeval/gt"
    fake_paths = r"/data1/liangshuaizhe/map_translate/202008_SH_toeval/noise"
    for layer in layers:
        for numresult in range(5):
            real_path = os.path.join(real_paths, layer)
            fake_path = os.path.join(fake_paths, str(numresult), layer)
            fid = fid_score(real_path=real_path, fake_path=fake_path, gpu='1')
            print(f'===> fid score:{fid:.4f}')
            target = os.path.join(target_path, 'noise_' + layer + '_' + str(numresult) + 'fid.json')
            with open(target, 'a') as f:
                json.dump(fid, f)

    # real
    real_paths = r"/data1/liangshuaizhe/map_translate/202008_SH_toeval/gt"
    fake_paths = ["/data1/liangshuaizhe/map_translate/202008_SH_toeval/1_layer_result",
                  "/data1/liangshuaizhe/map_translate/202008_SH_toeval/muti_layer_result/unbalance_200epoch",
                  "/data1/liangshuaizhe/map_translate/202008_SH_toeval/muti_layer_result/balance_70train_200epoch",
                  "/data1/liangshuaizhe/map_translate/202008_SH_toeval/muti_layer_result/balance_notest_50epoch"]

    for layer in layers:
        for numresult in range(len(fake_paths)):
            real_path = os.path.join(real_paths, layer)
            fake_path = os.path.join(fake_paths[numresult], layer)
            fid = fid_score(real_path=real_path, fake_path=fake_path, gpu='1')
            print(f'===> fid score:{fid:.4f}')
            target = os.path.join(target_path, 'real_' + layer + '_' + str(numresult) + 'fid.json')

            with open(target, 'a') as f:
                json.dump(fid, f)
