import os

opt_list = os.listdir("./output/group1")
for i in range(len(opt_list)):
    cmd = f"""splashsurf reconstruct -i ./output/group1/{i}.ply --output-dir=out_mesh/group1 --particle-radius=0.008 --smoothing-length=2.0 --cube-size=1.5 --surface-threshold=0.6 -o opt_{i}.obj"""
    os.system(cmd)
    # cmd = f"""splashsurf reconstruct -i ./output/group2/{i}.ply --output-dir=out_mesh/group2 --particle-radius=0.008 --smoothing-length=2.0 --cube-size=1.5 --surface-threshold=0.6 -o opt_{i}.obj"""
    # os.system(cmd)


opt_list = os.listdir("./out_mesh/group1")
for item in opt_list:
    index = int(item[4:-4])
    import sys
    os.rename(f"./out_mesh/group1/sequence_{index//5}.obj")
