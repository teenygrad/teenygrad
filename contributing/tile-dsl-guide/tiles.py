import math
F=4
def shp(B,CI,CO,H,W,K=3,S=1,P=1): return dict(B=B,CI=CI,CO=CO,H=H,W=W,K=K,S=S,P=P,OH=(H+2*P-K)//S+1,OW=(W+2*P-K)//S+1)
def tile(c,tb,tc,th,tw,rc):
    """output tile [tb,tc,th,tw]; rc = input channels staged per reduction step"""
    K,S=c['K'],c['S']
    ih=(th-1)*S+K; iw=(tw-1)*S+K
    n=math.ceil(c['B']/tb)*math.ceil(c['CO']/tc)*math.ceil(c['OH']/th)*math.ceil(c['OW']/tw)
    x_tr=tb*c['CI']*ih*iw; w_tr=tc*c['CI']*K*K; o=tb*tc*th*tw
    smem=(tb*rc*ih*iw + tc*rc*K*K)*F
    fused=n*(x_tr+w_tr+o+2*tc)*F
    return n,smem,fused
for name,c in [("test",shp(10,3,8,32,32)),("resnet",shp(32,64,64,56,56))]:
    outb=c['B']*c['CO']*c['OH']*c['OW']*F
    print("==",name,"out MB",outb/1e6,"x MB",c['B']*c['CI']*c['H']*c['W']*F/1e6,"w KB",c['CO']*c['CI']*9*F/1e3)
    for t in [(1,1,1,16),(1,1,1,32),(1,16,1,32),(1,32,1,32),(1,64,1,56),(1,32,8,8),(1,64,8,8),(1,64,4,28)]:
        rc=min(8,c['CI'])
        if t[1]>c['CO']: continue
        n,s,f=tile(c,*t,rc)
        print(f" tile={t} rc={rc} tiles={n:7d} smem={s:7d}B fused={f/1e6:9.2f}MB  (+bn,silu unfused add {4*outb/1e6:.1f}MB)")
