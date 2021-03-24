import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image
import astra
import scipy.sparse
import scipy.sparse.linalg
import time

%matplotlib inline


#def proj_A(x):
#    hp, hn, vp, vn = fht(x)


#def inv_A(hp, hn, vp, vn):
#    return ifht(hp, hn, vp, vn)


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


def fast_calc_grad1(Y, n=256, m=256):
    '''
    вычисляет градиент для матрицы ошибок
    params: 
    Y - текущий вектор из пикселей изображения
    n, m - размеры изображения
    return:
    TV-regularization gradient вектор
    '''
    Y = Y.reshape(-1)
    grad = np.zeros(n * m)
    idx1 = np.arange(m, n * m)
    idx2 = np.arange(n * m).reshape(n, m).T[1:].reshape(-1)
    grad[idx1] += 1 * ((Y[idx1] - Y[idx1 - m]) > 0) - 1 * ((Y[idx1] - Y[idx1 - m]) <= 0)
    grad[idx1 - m] += -1 * ((Y[idx1] - Y[idx1 - m]) > 0) + 1 * ((Y[idx1] - Y[idx1 - m]) <= 0)
    grad[idx2] += 1 * ((Y[idx2] - Y[idx2 - 1]) > 0) - 1 * ((Y[idx2] - Y[idx2 - 1]) <= 0)
    grad[idx2 - 1] += -1 * ((Y[idx2] - Y[idx2 - 1]) > 0) + 1 * ((Y[idx2] - Y[idx2 - 1]) <= 0)
    #print(grad.shape)
    return grad.reshape(-1, 1)


def fast_calc_grad_mu1(Y, mu=0.1, n=256, m=256):
    '''
    params: 
    Y - текущий вектор из пикселей изображения
    n, m - размеры изображения
    mu - коэффициент аппроксимации |x| ~ \sqrt(x^2 + mu)
    return:
    TV-regularization gradient
    '''
    Y = Y.reshape(-1)
    grad = np.zeros(n * m)
    idx1 = np.arange(m, n * m)
    idx2 = np.arange(n * m).reshape(n, m).T[1:].reshape(-1)
    abs_grad1 = (Y[idx1] - Y[idx1 - m]) / (((Y[idx1] - Y[idx1 - m]) ** 2 + mu) ** 0.5)
    abs_grad2 = (Y[idx2] - Y[idx2 - 1]) / (((Y[idx2] - Y[idx2 - 1]) ** 2 + mu) ** 0.5)
    grad[idx1] += (1 * ((Y[idx1] - Y[idx1 - m]) > 0) - 1 * ((Y[idx1] - Y[idx1 - m]) <= 0)) * abs_grad1
    grad[idx1 - m] += (-1 * ((Y[idx1] - Y[idx1 - m]) > 0) + 1 * ((Y[idx1] - Y[idx1 - m]) <= 0)) * abs_grad1
    grad[idx2] += (1 * ((Y[idx2] - Y[idx2 - 1]) > 0) - 1 * ((Y[idx2] - Y[idx2 - 1]) <= 0)) * abs_grad2
    grad[idx2 - 1] += (-1 * ((Y[idx2] - Y[idx2 - 1]) > 0) + 1 * ((Y[idx2] - Y[idx2 - 1]) <= 0)) * abs_grad2
    return grad.reshape(-1, 1)


def show_img1(x, iter, method, n=256, m=256):
    '''
    Show reconstructed image after iter iteration of method {SIRT, SIRT-reg, SIRT-reg-lambda}
    params:
    x - image
    n, m - image size
    iter - iteration number
    method - SIRT, SIRT-reg, SIRT-reg-lambda
    '''
    y = x.reshape((n, m))
    plt.imshow(y, cmap='gray')
    plt.title("{} after {} iter".format(method, iter))


def plot_sino1(sino):
    '''
    Visulise sinogram
    '''
    plt.imshow(sino, cmap='gray')


def opt_SIRT1(b, Y, angles, det_row_count, det_col_count, n, m):
    hp, hn, vp, vn = sino_to_lino(b, np.linspace(0, np.pi, angles) / np.pi * 180, det_col_cout)
    hp1, hn1, vp1, vn1 = fht(Y.reshape(n, m))
    return np.linalg.norm(np.concatenate((hp - hp1, hn - hn1, vp - vp1, vn - vn1)))
    #return np.linalg.norm(b - A.dot(Y))


def opt_SIRT_reg1(Y, n, m, alpha):
    #syst = np.linalg.norm(b - A.dot(Y))
    Y = Y.reshape(-1)
    idx1 = np.arange(m, n * m)
    idx2 = np.arange(n * m).reshape(n, m).T[1:].reshape(-1)
    total_variation = alpha * (np.sum(np.abs(Y[idx1] - Y[idx1 - m])) + np.sum(np.abs(Y[idx2] - Y[idx2 - 1])))
    return total_variation


#def opt_SIRT_reg_mu1(A, b, Y, n, m, alpha, mu):
#    syst = np.linalg.norm(b - A.dot(Y))
#    Y = Y.reshape(-1)
#    idx1 = np.arange(m, n * m)
#    idx2 = np.arange(n * m).reshape(n, m).T[1:].reshape(-1)
#    total_variation_mu = alpha * (np.sum(((Y[idx1] - Y[idx1 - m]) ** 2 + mu) ** 0.5) + np.sum(((Y[idx2] - Y[idx2 - 1]) ** 2 + mu) ** 0.5))
#    return syst + total_variation_mu


def grad_SIRT1(b, Y, angles, det_row_count, det_col_count, n, m):
    #print("?????")
    hp, hn, vp, vn = fht(Y.reshape(n, m))
    #print(hp, hn, vp, vn)
    #print("*****")
    hp1, hn1, vp1, vn1 = sino_to_lino(b, np.linspace(0, np.pi, angles) / np.pi * 180, det_col_count)
    #print("&&&&&")
    x = ifht(hp1 - hp, hn1 - hn, vp1 - vp, vn1 - vn)
    #print("@@@@@")
    x = -x[::-1][::-1]
    return x.reshape(-1, 1)
    #return -backproj_matr.dot(b - A.dot(Y))


def grad_SIRT_reg1(Y, n, m, alpha):
    return alpha * fast_calc_grad1(Y, n, m)


#def grad_SIRT_reg_mu1(backproj_matr, A, b, Y, n, m, alpha, mu):
#    return -backproj_matr.dot(b - A.dot(Y)) + alpha * fast_calc_grad_mu(Y, mu, n, m)


def find_optimal_lambda_SIRT1(b, Y, angles, det_row_count, det_col_count, n, m):
    l = -1000
    r = 1000
    grad1 = grad_SIRT1(b, Y, angles, det_row_count, det_col_count, n, m)
    hp, hn, vp, vn = sino_to_lino(b, np.linspace(0, np.pi, angles) / np.pi * 180, det_col_count)
    hp1, hn1, vp1, vn1 = fht(Y.reshape(n, m))
    hp2, hn2, vp2, vn2 = fht(grad1.reshape(n, m))

    for i in range(40):
        m1 = l + (r - l) / 3
        m2 = r - (r - l) / 3
        #grad1 = grad_SIRT1(b, Y, n, m)
        #val1 = opt_SIRT1(b, Y - m1 * grad1, n, m)
        #grad2 = grad_SIRT1(b, Y, n, m)
        #val2 = opt_SIRT1(b, Y - m2 * grad1, n, m)
        val1 = np.linalg.norm(np.concatenate((hp - (hp1 - m1 * hp2), hn - (hn1 - m1 * hn2), 
                                              vp - (vp1 - m1 * vp2), vn - (vn1 - m1 * vn2))))
        val2 = np.linalg.norm(np.concatenate((hp - (hp1 - m2 * hp2), hn - (hn1 - m2 * hn2), 
                                              vp - (vp1 - m2 * vp2), vn - (vn1 - m2 * vn2))))
        #val1 += opt_SIRT_reg1(Y - m1 * grad1, n, m, alpha)
        #val2 += opt_SIRT_reg1(Y - m2 * grad1, n, m, alpha)
        if val1 < val2:
            r = m2
        else:
            l = m1
    return l


def find_optimal_lambda_SIRT_reg1(b, Y, angles, det_row_count, det_col_count, n, m, alpha):
    l = -1000
    r = 1000
    grad1 = grad_SIRT1(b, Y, angles, det_row_count, det_col_count, n, m) + grad_SIRT_reg1(Y, n, m, alpha)
    hp, hn, vp, vn = sino_to_lino(b, np.linspace(0, np.pi, angles) / np.pi * 180, det_col_count)
    hp1, hn1, vp1, vn1 = fht(Y.reshape(n, m))
    hp2, hn2, vp2, vn2 = fht(grad1.reshape(n, m))
    for i in range(40):
        m1 = l + (r - l) / 3
        m2 = r - (r - l) / 3
        val1 = np.linalg.norm(np.concatenate((hp - (hp1 - m1 * hp2), hn - (hn1 - m1 * hn2), 
                                              vp - (vp1 - m1 * vp2), vn - (vn1 - m1 * vn2)))) ** 2
        val2 = np.linalg.norm(np.concatenate((hp - (hp1 - m2 * hp2), hn - (hn1 - m2 * hn2), 
                                              vp - (vp1 - m2 * vp2), vn - (vn1 - m2 * vn2)))) ** 2
        val1 += opt_SIRT_reg1(Y - m1 * grad1, n, m, alpha)
        val2 += opt_SIRT_reg1(Y - m2 * grad1, n, m, alpha)
        
        #grad1 = grad_SIRT_reg(backproj_matr, A, b, Y, n, m, alpha)
        #val1 = opt_SIRT_reg(A, b, Y - m1 * grad1, n, m, alpha)
        #grad2 = grad_SIRT_reg(backproj_matr, A, b, Y, n, m, alpha)
        #val2 = opt_SIRT_reg(A, b, Y - m2 * grad2, n, m, alpha)
        if val1 < val2:
            r = m2
        else:
            l = m1
    return l


#def find_optimal_lambda_SIRT_reg_mu1(backproj_matr, A, b, Y, n, m, alpha, mu):
#    l = -1000
#    r = 1000
#    for i in range(40):
#        m1 = l + (r - l) / 3
#        m2 = r - (r - l) / 3
#        grad1 = grad_SIRT_reg_mu(backproj_matr, A, b, Y, n, m, alpha, mu)
#        val1 = opt_SIRT_reg_mu(A, b, Y - m1 * grad1, n, m, alpha, mu)
#        grad2 = grad_SIRT_reg_mu(backproj_matr, A, b, Y, n, m, alpha, mu)
#        val2 = opt_SIRT_reg_mu(A, b, Y - m2 * grad2, n, m, alpha, mu)
#        if val1 < val2:
#            r = m2
#        else:
#            l = m1
#    return l


def SIRT1(b, angles, det_row_count, det_col_count, n=256, m=256, iter=200, threshold=1e-4, show_plots=True):
    '''
    params:
    A - матрица системы линейных уравнений(sparse-matrix)
    b - вектор проекций(sparse-matrix)
    n, m - размер изображения
    iter - количество итераций
    return:
    вектор пикселей изображения, 
    '''
    start = time.time()
    x = np.zeros((n * m, 1))
    for i in range(iter):
        y = x
        alpha = find_optimal_lambda_SIRT1(b.T, x, angles, det_row_count, det_col_count, n, m)
        print(alpha)
        x = x - alpha * grad_SIRT1(b.T, x, angles, det_row_count, det_col_count, n, m)
        print(sum(x))
        if np.linalg.norm(y - x) < threshold:
            break
        if i % 1 == 0 and show_plots:
            show_img1(x[::-1, ::-1], i, 'SIRT')
            plt.show()
    end = time.time()
    show_img1(x[::-1, ::-1], iter, 'SIRT')
    plt.show()
    print('Время работы SIRT: {} секунд'.format(end - start))
    return x


def SIRT_reg1(b, angles, det_row_count, det_col_count, n=256, m=256, alpha=11, iter=200, threshold=1e-4, show_plots=True):
    '''
    params:
    A - матрица системы линейных уравнений(sparse-matrix)
    b - вектор проекций(sparse-matrix)
    n, m - размер изображения
    alpha - коэффициент регуляризации
    iter - количество итераций
    return:
    вектор пикселей изображения
    '''
    start = time.time()
    x = np.zeros((n * m, 1))
    for i in range(iter):
        y = x
        coef = find_optimal_lambda_SIRT_reg1(b.T, x, angles, det_row_count, det_col_count, n, m, alpha)
        x = x - coef * grad_SIRT1(b.T, x, angles, det_row_count, det_col_count, n, m)
        x = x + coef * (-alpha) * fast_calc_grad1(y)
        if np.linalg.norm(y - x) < threshold:
            break
        if i % 1 == 0 and show_plots:
            show_img1(x[::-1, ::-1], i, 'SIRT-reg')
            plt.show()
    end = time.time()
    show_img(x, iter, 'SIRT-reg')
    plt.show()
    print('Время работы SIRT с TV-регуляризацией: {} секунд'.format(end - start))
    return x


def SIRT_reg_mu1(A, b, n=256, m=256, alpha=0.01, mu=0.01, iter=200, threshold=1e-4, show_plots=False):
    '''
    params:
    A - матрица системы линейных уравнений(sparse-matrix)
    b - вектор проекций(sparse-matrix)
    n, m - размер изображения
    alpha - коэффициент регуляризации
    mu - коэффициент сглажиания
    iter - количество итераций
    return:
    вектор пикселей изображения
    '''
    start = time.time()
    C1 = np.array(1 / scipy.sparse.csr_matrix.sum(A, axis=0)).reshape(-1)
    R1 = np.array(1 / scipy.sparse.csr_matrix.sum(A, axis=1)).reshape(-1)
    elem = [np.arange(C1.shape[0]), np.arange(C1.shape[0])]
    elem1 = [range(R1.shape[0]), range(R1.shape[0])]
    C = scipy.sparse.csr_matrix((C1, (elem[0], elem[1])), shape=(len(C1), len(C1)))
    R = scipy.sparse.csr_matrix((R1, (elem1[0], elem1[1])), shape=(len(R1), len(R1)))
    b = b.reshape(-1, 1)
    x = np.zeros((n * m, 1))
    backproj_matr = (C.dot(A.T)).dot(R)
    #x = np.zeros((256 * 256))
        #x += scipy.sparse.csc_matrix.dot(scipy.sparse.csc_matrix.dot(scipy.sparse.csc_matrix.dot(C, A.T), R), b - scipy.sparse.csc_matrix.dot(A, scipy.sparse.csc_matrix(x)))
    #x += -alpha * calc_grad_mu(x, 0.1)
    for i in range(iter):
        #lamb = np.argmin(np.linspace(-1, 1, 1000))
        y = x
        coef = find_optimal_lambda_SIRT_reg_mu(backproj_matr, A, b, x, n, m, alpha, mu)
        #backproj_matr = (C.dot(A.T)).dot(R)
        x = x + coef * backproj_matr.dot(b - A.dot(x))
        x += coef * (-alpha) * fast_calc_grad_mu(x, mu)
        if np.linalg.norm(y - x) < threshold:
            break
        if i % 10 == 0 and show_plots:
            show_img(x, i, 'SIRT-reg-mu')
            plt.show()
    end = time.time()
    show_img(x, iter, 'SIRT-reg-mu')
    plt.show()
    print('Время работы SIRT с TV-регуляризацией и сглаживанием: {} секунд'.format(end - start))
    return x


def get_A_b(img, angles=18, det_row_count=1, det_col_count=256):
    '''
    Возвращает матрицу системы линейных уравнений
    params:
    img - изображение
    angles - количество углов проекций
    det_row_count - расстояние между 2 проекциями детектора
    det_col_count - количество измерений для каждого угла
    return:
    A - system matrix
    b - projection vector
    '''
    vol_geom = astra.create_vol_geom(img.shape[0], img.shape[1])
    angles = np.linspace(0, np.pi, angles)
    proj_geom = astra.create_proj_geom('parallel', det_row_count, det_col_count, angles)
    proj_id = astra.create_projector('linear', proj_geom, vol_geom)
    matrix_id = astra.projector.matrix(proj_id)
    A = astra.matrix.get(matrix_id).astype(np.float)
    sinogram_id, sino = astra.create_sino(img, proj_id)
    return A, sino


def prepare_image1(img, n, m):
    '''
    Resize image to nxm and gray-scale it
    '''
    img = img.resize((n, m))
    img = np.asarray(img)
    img = rgb2gray(img)
    return img


def SIRT_libr1(img, angles=18, det_row_count=1, det_col_count=256):
    '''
    Стандартный библиотечный SIRT
    '''
    vol_geom = astra.create_vol_geom(img.shape[0], img.shape[1])
    angles = np.linspace(0, np.pi, angles)
    proj_geom = astra.create_proj_geom('parallel', det_row_count, det_col_count, angles)
    proj_id = astra.create_projector('linear', proj_geom, vol_geom)
    sinogram_id, sino = astra.create_sino(img, proj_id)
    start = time.time()
    recon_id = astra.data2d.create('-vol', vol_geom, 0)
    cfg = astra.astra_dict('SIRT')
    cfg['ProjectorId'] = proj_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = recon_id
    algorithm_id = astra.algorithm.create(cfg)
    astra.algorithm.run(algorithm_id, 200)
    new_img = astra.data2d.get(recon_id)
    end = time.time()
    plt.title('Стандартный SIRT')
    plt.imshow(new_img, cmap='gray')
    plt.show()
    print('Время работы SIRT из стандартной библиотеки: {} секунд'.format(end - start))


def phantom_test1(angles=18, det_row_count=1, det_col_count=256, alpha=1, mu=0.1):
    img = Image.open('drive/MyDrive/Диплом/SheppLogan_Phantom.png') #path to image
    img = prepare_image1(img, 256, 256)
    print(img.min(), img.max())
    A, b = get_A_b(img, angles, det_row_count, det_col_count)
    plt.title('Фантом')
    plt.imshow(img, cmap='gray')
    plt.show()
    plt.title('Синограмма')
    plot_sino1(b)
    plt.show()
    #SIRT_libr(img)
    #SIRT1(b, angles, det_row_count, det_col_count)
    SIRT_reg1(b, angles, det_row_count, det_col_count)
    #SIRT_reg_mu(A, b)
