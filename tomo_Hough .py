#encoding: utf-8
from __future__ import division
import numpy as np

def sino_to_lino(sino, ang, N, r=5):
    """
    Преобразование синограммы в четыре составляющих линограммы.  
    :param sino: Исходная синограмма
    :param ang: Углы измерения (в градусах)
    :param N: Линейный размер реконструируемого изображения
    :param r: Отклонение от границы сектора линогаммы, учитываемое при интерполяции 
    данных
    """
    if r:
        sino = np.pad(sino, ((0, 0), (r, r)), mode='reflect')
        ang = np.hstack((-ang[r:0:-1], ang, -ang[-r::1]))
        index_hp = (ang >= 45 - r) & (ang <= 90 + r)
        index_hn = (ang >= 90 - r) & (ang <= 135 + r)
        index_vp = (ang >= 0) & (ang <= 45 + r)
        index_vn = (ang >= 135 - r) & (ang <= 180 + r)
    else:
        raise ValueError('r must be grate than 0, given {}'.format(r))
    
    pn = sino.shape[0]
    r = np.arange(-pn // 2, pn // 2)

    # hp
    index = index_hp
    sino_hp = sino[:, index]
    sino_new = np.zeros((2 * N, sino_hp.shape[1]))
    t = N / np.tan(np.deg2rad(ang[index]))
    for j in np.arange(sino_hp.shape[1]):
        sino_new[:, j] = np.interp(np.arange(-N, N), r * np.sqrt(t[j] ** 2 + N ** 2) / N + N / 2 - t[j] / 2 + 1,
                                   sino_hp[:, j])
    hp = np.zeros((2 * N, N))
    for i in np.arange(2 * N):
        hp[i - N, :] = np.interp(np.arange(0, N), t[::-1], sino_new[i, ::-1])

    # hn
    index = index_hn
    sino_hn = sino[:, index]
    sino_new = np.zeros((2 * N, sino_hn.shape[1]))
    t = N / np.tan(np.deg2rad(ang[index]))
    for j in np.arange(sino_hn.shape[1]):
        sino_new[:, j] = np.interp(np.arange(-N, N), r * np.sqrt(t[j] ** 2 + N ** 2) / N - t[j] / 2 - N / 2 + 1,
                                   sino_hn[:, j])
    hn = np.zeros((2 * N, N))
    for i in np.arange(2 * N):
        hn[i, :] = np.interp(np.arange(0, N), -t, sino_new[i, :])

    # vp
    index = index_vp
    sino_vp = sino[:, index]
    sino_new = np.zeros((2 * N, sino_vp.shape[1]))
    t = N * np.tan(np.deg2rad(ang[index]))
    for j in np.arange(sino_vp.shape[1]):
        sino_new[:, j] = np.interp(np.arange(-N, N), (r * np.sqrt(t[j] ** 2 + N ** 2) / N - N / 2 + t[j] / 2),
                                   sino_vp[:, j])
    vp = np.zeros((2 * N, N))
    for i in np.arange(2 * N):
        vp[i - N, :] = np.interp(np.arange(0, N), t, sino_new[i, :])
    vp = vp[::-1, :]

    # vn
    index = index_vn
    sino_vn = sino[:, index]
    sino_new = np.zeros((2 * N, sino_vn.shape[1]))
    t = N * np.tan(np.deg2rad(ang[index]))
    for j in np.arange(sino_vn.shape[1]):
        sino_new[:, j] = np.interp(np.arange(-N, N) + 1,
                                   -(r * np.sqrt(t[j] ** 2 + N ** 2) / N + N / 2 + t[j] / 2)[::-1], sino_vn[:, j])
    vn = np.zeros((2 * N, N))
    for i in np.arange(2 * N):
        vn[i, :] = np.interp(np.arange(0, N), -t[::-1], sino_new[i, ::-1])

    k = 1 + np.arange(N) ** 2 / (N - 1) ** 2
    return hp / k, hn / k, vp / k, vn / k


   def fht2(im, sign):
    """
    Вычислние быстрого преобразования Хафа для преимущественно горизонтальных прямых. Для нахождени БПХ для
    преимущественно вертикальных прямых используется fht2(im.T, sign). Подробное описание
    Ершов, Е. И., Терехин, А. П., & Николаев, Д. П. (2017).
    Обобщение быстрого преобразования Хафа для трехмерных изображений.
    Информационные процессы, 17(4), 294-308.
    :param im: исходное изображение
    :param sign: 1 для прямых с уклон вверх, -1 - для прямых с уклоном вниз
    """
    m, n = im.shape
    n0 = int(np.round(n / 2))
    if n < 2:
        h = im
    else:
        h = mergeHT(fht2(im[:, 0:n0], sign), fht2(im[:, n0::], sign), sign)
    return h


def mergeHT(h0, h1, sign):
    """
    Вспомогательная функция вычисления БПХ.
    """
    m, n0 = h0.shape
    n = 2 * n0
    h = np.zeros((m, n))
    r0 = (n0 - 1) / (n - 1)
    for t in np.arange(n):
        t0 = int(np.around(t * r0))
        s = int(sign * (t - t0))
        h[:, t] = h0[:, t0] + np.hstack((h1[s:m, t0], h1[0:s, t0]))
    return h


def fht(im):
    """
    Вычислние быстрого преобразования Хафа для всех четырех типов прямых. 
    Подробное описание
    Ершов, Е. И., Терехин, А. П., & Николаев, Д. П. (2017).
    Важно! Для точного восстановления необходимо, чтобы Хаф-образы сожержали полные диапазон 
    углов (t=-N..N-1), а не были рекурсивно замкнуты, как в случае применения функции fht2.
    """
    N = im.shape[0]

    im_pad_h = np.pad(im, ((0, N), (0, 0)), mode='constant')
    im_pad_v = np.pad(im, ((0, 0), (0, N)), mode='constant')
    k = np.sqrt(1 + np.arange(N) ** 2 / (N - 1) ** 2)

    hp = fht2(im_pad_h, 1)
    hn = fht2(im_pad_h, -1)
    vp = fht2(im_pad_v.T, 1)
    vn = fht2(im_pad_v.T, -1)
    return hp, hn, vp, vn


def ifht(hp, hn, vp, vn):
    """
    Обратное проецирование с помощью БПХ. Входные данные в формате БПХ. Для применения к синограмме необходимо
    предварительно выполнить смену координат и интерполяцию для приведения к нужной сетке.
    (Ершов, Е. И., Терехин, А. П., & Николаев, Д. П. (2017).
    Обобщение быстрого преобразования Хафа для трехмерных изображений.
    Информационные процессы, 17(4), 294-308.)
    Важно! Для точного восстановления необходимо, чтобы Хаф-образы сожержали полные диапазон 
    углов (t=-N..N-1), а не были рекурсивно замкнуты, как в случае применения функции fht2.
    :param hp: БПХ для преимущественно горизонтальных прямых с уклоном вверх
    :param hn: БПХ для преимущественно горизонтальных прямых с уклоном вниз
    :param vp: БПХ для преимущественно вертикальных прямых с уклоном вправо
    :param vn: БПХ для преимущественно вертикальных прямых с уклоном влево
    """
    N = hp.shape[1]

    hpi = fht2(hp, -1)
    hni = fht2(hn, 1)
    vpi = fht2(vp, -1).T
    vni = fht2(vn, 1).T

    return (hpi[:N, :N] + hni[:N, :N] + vpi[:N, :N] + vni[:N, :N]) / N