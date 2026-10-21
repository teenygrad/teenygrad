import math
F=4
def cfg(name,B,CI,CO,H,W,K=3,S=1,P=1,G=1):
    OH=(H+2*P-K)//S+1; OW=(W+2*P-K)//S+1
    return dict(name=name,B=B,CI=CI,CO=CO,H=H,W=W,K=K,S=S,P=P,G=G,OH=OH,OW=OW)
def per_tile(c,BO,declared=False):
    B,CI,CO,H,W,K,S,G,OH,OW=[c[k] for k in "B CI CO H W K S G OH OW".split()]
    bo=min(BO,OW)  # elements actually live; tile width BO but extent caps
    span=(BO-1)*S+K
    span_c=min(span,W+2*c['P'])
    if declared:
        x=B*CI*H*span_c   # B,C_IN,H unresolved -> full extent ; W -> receptive field
        w=0
    else:
        x=1*(CI//G)*K*span_c
        w=(CI//G)*K*K
    out=BO
    return x,w,out
def run(c):
    print("==",c['name'],{k:c[k] for k in "B CI CO H W OH OW".split()})
    n_out=c['B']*c['CO']*c['OH']*c['OW']
    print(" output elems",n_out,"bytes",n_out*F)
    for BO in [8,16,32,64]:
        if BO>2*c['OW'] : continue
        ntiles=c['B']*c['CO']*c['OH']*math.ceil(c['OW']/BO)
        x,w,o=per_tile(c,BO)
        xd,_,_=per_tile(c,BO,True)
        fused_tile=(x+w+2+o)*F      # bn folded scale/shift = 2 scalars
        conv_tile=(x+w+o)*F
        fused=ntiles*fused_tile
        unfused=ntiles*conv_tile + 2*n_out*F + 2*c['CO']*F + 2*n_out*F  # conv ; bn r/w + params(scale,shift) ; silu r/w
        unfused_bn4 = ntiles*conv_tile + 2*n_out*F + 4*c['CO']*F + 2*n_out*F
        print(f" BO={BO:3d} tiles={ntiles:7d} xreg={x:5d} w={w:4d} footprint(x+w)={ (x+w)*F:7d}B declared_x={xd*F:9d}B  fused={fused/1e6:8.3f}MB unfused={unfused/1e6:8.3f}MB unfused(bn4)={unfused_bn4/1e6:8.3f}MB ratio={unfused/fused:5.2f}")
    slab=c['B']*c['OH']*c['OW']*F
    print(" training channel slab bytes", slab, " KB",slab/1024)
for c in [cfg("test",10,3,8,32,32), cfg("resnet",32,64,64,56,56)]:
    run(c)
