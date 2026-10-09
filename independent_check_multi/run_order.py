from msim import init_scene, run_trial
print("Sensitivity: velocity-first update (v<-v+u; p<-p+v), S=60, 100 trials, % all captured")
for N in (4,8):
    for vs in (0,5):
        for M in (3,5):
            cid = N*1000+M*100+60+vs*7; row=[]
            for meth in ('gwo_avg','gwo_group','hungarian','bottleneck'):
                ok=sum(run_trial(meth,*init_scene(cid*100000+t,N,M,60,vs),seed=5_000_000+cid*1000+t,order='vp')['T'] is not None for t in range(100))
                row.append(f"{meth}={ok:3d}")
            print(f"N={N} vt={vs} M={M}: "+"  ".join(row))
