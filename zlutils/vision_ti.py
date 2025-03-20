import taichi as ti
import taichi.math as tm
import numpy as np
import cv2


ti.init(arch=ti.cpu)


@ti.kernel
def _mixRGB(
    imat: ti.types.ndarray(dtype=ti.uint8),
    wr: ti.f32,
    wg: ti.f32,
    wb: ti.f32,
    out: ti.types.ndarray(dtype=ti.uint8)
):
    for y, x in out:
        out[y, x] = ti.floor(
            imat[y, x, 0]*wb + imat[y, x, 1]*wg + imat[y, x, 2]*wr,
            dtype=ti.uint8
            )


def mixRGB(imat, wr=0.299, wg=0.587, wb=0.114):
    #  根据权重混合rgb
    #  默认rgb权重 0.299, 0.587, 0.114
    out = np.zeros(imat.shape[:2], dtype=np.uint8)
    _mixRGB(imat, wr, wg, wb, out)
    return out


def mixBGR(imat, wb=0.114, wg=0.587, wr=0.299):
    imat = cv2.cvtColor(imat, cv2.COLOR_BGR2RGB)
    return mixRGB(imat, wr=wr, wg=wg, wb=wb)


@ti.func
def __absDiff(x, y, dtype=ti.u8):
    r = dtype(0)
    if x > y:
        r = x-y
    else:
        r = y-x
    return r


@ti.kernel
def _mixBaseRGB_rough(
    imat: ti.types.ndarray(dtype=ti.uint8),
    r: ti.u8,
    g: ti.u8,
    b: ti.u8,
    wr: ti.f32,
    wg: ti.f32,
    wb: ti.f32,
    out: ti.types.ndarray(dtype=ti.uint8)
):
    for y, x in out:
        out[y, x] = ti.floor(
            255.
            - __absDiff(imat[y, x, 0], r)*wb
            - __absDiff(imat[y, x, 1], g)*wg
            - __absDiff(imat[y, x, 2], b)*wr,
            dtype=ti.u8
            )


@ti.kernel
def _mixBaseRGB(
    imat: ti.types.ndarray(dtype=ti.uint8),
    r: ti.u8,
    g: ti.u8,
    b: ti.u8,
    wr: ti.f32,
    wg: ti.f32,
    wb: ti.f32,
    out: ti.types.ndarray(dtype=ti.uint8)
):  # 按比例放缩权重
    rl = 255./(r-0)*wr
    rr = 255./(255-r)*wr
    wwr = ti.min(rl, rr)
    gl = 255./(g-0)*wg
    gr = 255./(255-g)*wg
    wwg = ti.min(gl, gr)
    bl = 255./(b-0)*wb
    br = 255./(255-b)*wb
    wwb = ti.min(bl, br)
    for y, x in out:
        rst = ti.f32(255.)
        if imat[y, x, 0] > r:
            rst = rst - (imat[y, x, 0] - r)*wwr
        else:
            rst = rst - (r - imat[y, x, 0])*wwr
        if imat[y, x, 1] > g:
            rst = rst - (imat[y, x, 1] - g)*wwg
        else:
            rst = rst - (g - imat[y, x, 1])*wwg
        if imat[y, x, 2] > b:
            rst = rst - (imat[y, x, 2] - b)*wwb
        else:
            rst = rst - (b - imat[y, x, 2])*wwb
        out[y, x] = ti.floor(rst, ti.u8)


def mixBaseRGB(imat, r=255, g=255, b=255, wr=0.299, wg=0.587, wb=0.114, rough=False):
    # 基于到指定rgb颜色的距离和权重混合成灰度图
    # 距离越接近颜色越白
    # rough 是否放缩权重
    out = np.zeros(imat.shape[:2], dtype=np.uint8)
    imat = np.ascontiguousarray(imat)
    if rough:
        _mixBaseRGB_rough(imat=imat, r=r, g=g, b=b, wr=wr, wg=wg, wb=wb, out=out)
    else:
        _mixBaseRGB(imat=imat, r=r, g=g, b=b, wr=wr, wg=wg, wb=wb, out=out)
    return out


@ti.kernel
def _quantilize8BitRGB(
    imat: ti.types.ndarray(dtype=ti.uint8),
    wr: ti.f32,
    wg: ti.f32,
    wb: ti.f32,
    out: ti.types.ndarray(dtype=ti.uint8)
):
    # 量化24位颜色到8位，与灰度化不同，量化后的像素值不代表深度
    # TODO
    pass


@ti.kernel
def _maxPoolingGrey(
    gmat: ti.types.ndarray(dtype=ti.uint8),
    pool_size: ti.i32,
    strides: ti.i32,
    pix_skip: ti.i32,
    out: ti.types.ndarray(dtype=ti.uint8),
):
    for h, w in out:
        max_val = ti.uint8(0)
        hs = h*strides
        ws = w*strides
        for i in range(0, pool_size//pix_skip):
            for j in range(0, pool_size//pix_skip):
                val = gmat[hs + i*pix_skip, ws + j*pix_skip]
                if val > max_val:
                    max_val = val
        out[h, w] = max_val


def maxPoolingGrey(gmat, pool_size=8, strides=None, pix_skip=1):
    gmat = np.ascontiguousarray(gmat)
    if strides is None:
        strides = pool_size
    pooled_height = (gmat.shape[0] - strides) // strides + 1
    pooled_width = (gmat.shape[1] - strides) // strides + 1
    out = np.zeros((pooled_height, pooled_width), dtype=np.uint8)
    _maxPoolingGrey(gmat=gmat, pool_size=pool_size, strides=strides, pix_skip=pix_skip, out=out)
    return out


@ti.kernel
def _pipeBmatFeature(
    imat: ti.types.ndarray(dtype=ti.uint8),
    r: ti.u8,
    g: ti.u8,
    b: ti.u8,
    wr: ti.f32,
    wg: ti.f32,
    wb: ti.f32,
    pool_size: ti.i32,
    strides: ti.i32,
    pix_skip: ti.i32,
    bthresh: ti.f32,
    out: ti.types.ndarray(dtype=ti.u8)
):
    wrgb = ti.Vector([wr, wg, wb], dt=ti.f32)
    rgb = ti.Vector([r, g, b], dt=ti.u8)
    for ci in range(3):
        wrgb[ci] = ti.min(255./(rgb[ci]-0)*wrgb[ci], 255./(255-rgb[ci])*wrgb[ci])
    ibthresh = ti.f32(255.) - bthresh
    m = pool_size//pix_skip
    n = pool_size//pix_skip

    for h, w in out:
        bflag = ti.u8(0)
        hs = h*strides
        ws = w*strides
        for i, j in ti.ndrange(m, n):
            # rst = ti.f32(255.)
            y = hs + i*pix_skip
            x = ws + j*pix_skip
            rstv = ti.Vector([0., 0., 0.], dt=ti.f32)
            for ci in range(3):
                if imat[y, x, ci] > rgb[ci]:
                    rstv[ci] = (imat[y, x, ci] - rgb[ci])*wrgb[ci]
                else:
                    rstv[ci] = (rgb[ci] - imat[y, x, ci])*wrgb[ci]
            if (rstv[0]+rstv[1]+rstv[2]) <= ibthresh:
                bflag = ti.u8(255)
                break

        out[h, w] = bflag


@ti.kernel
def _pipeBmatFeature_rough(
    imat: ti.types.ndarray(dtype=ti.uint8),
    r: ti.u8,
    g: ti.u8,
    b: ti.u8,
    pool_size: ti.i32,
    strides: ti.i32,
    pix_skip: ti.i32,
    bthresh: ti.f32,
    out: ti.types.ndarray(dtype=ti.u8)
    # out: ti.types.ndarray(dtype=ti.u1),
):
    m = pool_size//pix_skip
    n = pool_size//pix_skip
    wbthresh = bthresh*3

    for h, w in out:
        bflag = ti.u8(0)
        hs = h*strides
        ws = w*strides
        for i, j in ti.ndrange(m, n):
            rst = ti.f32(255.*3)
            y = hs + i*pix_skip
            x = ws + j*pix_skip
            if imat[y, x, 0] > r:
                rst = rst - (imat[y, x, 0] - r)
            else:
                rst = rst - (r - imat[y, x, 0])
            if imat[y, x, 1] > g:
                rst = rst - (imat[y, x, 1] - g)
            else:
                rst = rst - (g - imat[y, x, 1])
            if imat[y, x, 2] > b:
                rst = rst - (imat[y, x, 2] - b)
            else:
                rst = rst - (b - imat[y, x, 2])
            if rst > wbthresh:
                bflag = ti.u8(255)
                break

        out[h, w] = bflag


def pipeBmatFeature(
    imat,
    r=255, g=255, b=255, wr=0.299, wg=0.587, wb=0.114,
    pool_size=8,
    strides=None,
    pix_skip=1,
    bthresh=int(256*0.85),
    rough=False
):
    # 输入rbg图片以及基准rgb颜色，输出对应的二维特征（取值0或255,dtype=np.uint8）
    # rough rgb按固定1：1：1
    imat = np.ascontiguousarray(imat)
    if strides is None:
        strides = pool_size
    pooled_height = (imat.shape[0] - strides) // strides + 1
    pooled_width = (imat.shape[1] - strides) // strides + 1
    out = np.zeros((pooled_height, pooled_width), dtype=np.uint8)
    if rough:
        _pipeBmatFeature_rough(
            imat,
            r=r, g=g, b=b,
            pool_size=pool_size,
            strides=strides,
            pix_skip=pix_skip,
            bthresh=bthresh,
            out=out,
        )
    else:
        _pipeBmatFeature(
            imat,
            r=r, g=g, b=b, wr=wr, wg=wg, wb=wb,
            pool_size=pool_size,
            strides=strides,
            pix_skip=pix_skip,
            bthresh=bthresh,
            out=out,
        )
    return out


@ti.kernel
def _pipeGmatFeature(
    imat: ti.types.ndarray(dtype=ti.uint8),
    r: ti.u8,
    g: ti.u8,
    b: ti.u8,
    pool_size: ti.i32,
    strides: ti.i32,
    pix_skip: ti.i32,
    outc: ti.types.ndarray(dtype=ti.u8),
):
    rgb = ti.Vector([r, g, b], dt=ti.u8)
    m = pool_size
    n = pool_size
    if pool_size > pix_skip:
        m = pool_size//pix_skip
        n = pool_size//pix_skip

    for h, w, c in outc:
        min_val = ti.u8(255)
        hs = h*strides
        ws = w*strides
        for i, j in ti.ndrange(m, n):
            y = hs + i*pix_skip
            x = ws + j*pix_skip
            val = ti.u8(0)
            if imat[y, x, c] > rgb[c]:
                val = (imat[y, x, c] - rgb[c])
            else:
                val = (rgb[c] - imat[y, x, c])
            if val < min_val:
                min_val = val
        outc[h, w, c] = ti.u8(255) - min_val


def pipeGmatFeature(
    imat,
    r=255, g=255, b=255,
    pool_size=8,
    strides=None,
    pix_skip=1,
    rough=False
):
    # 输入rbg图片以及基准rgb颜色，输出对应的二维特征（取值0~255,dtype=np.uint8）
    # 不进行二值化
    imat = np.ascontiguousarray(imat)
    if strides is None:
        strides = pool_size
    pooled_height = (imat.shape[0] - strides) // strides + 1
    pooled_width = (imat.shape[1] - strides) // strides + 1
    outc = np.zeros((pooled_height, pooled_width, 3), dtype=np.uint8)
    _pipeGmatFeature(
        imat,
        r=r, g=g, b=b,
        pool_size=pool_size,
        strides=strides,
        pix_skip=pix_skip,
        outc=outc,
    )
    return cv2.cvtColor(outc, cv2.COLOR_RGB2GRAY)
