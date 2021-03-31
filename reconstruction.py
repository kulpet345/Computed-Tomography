import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image
import astra
import scipy.sparse
import scipy.sparse.linalg
import time

%matplotlib inline


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

def fast_calc_grad(Y, n=256, m=256):
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
    return grad.reshape(-1, 1)


def grad_SIRT1(b, Y, angles, det_row_count, det_col_count, n, m):
    '''
    Возвращает градиент от матрицы со среднеквадратичной ошибкой
    '''
    hp, hn, vp, vn = fht(Y.reshape(n, m))
    hp1, hn1, vp1, vn1 = sino_to_lino(b, angles / np.pi * 180, det_col_count)
    x = ifht(hp1 - hp, hn1 - hn, vp1 - vp, vn1 - vn)
    x = -x[::-1][::-1]
    return x.reshape(-1, 1)


def TV_SIRT_reg1(Y, n, m, alpha):
    '''
    Считает значение TV-регуляризатора
    '''
    Y = Y.reshape(-1)
    idx1 = np.arange(m, n * m)
    idx2 = np.arange(n * m).reshape(n, m).T[1:].reshape(-1)
    total_variation = alpha * (np.sum(np.abs(Y[idx1] - Y[idx1 - m])) + np.sum(np.abs(Y[idx2] - Y[idx2 - 1])))
    return total_variation


def show_img(x, iter, method, n=256, m=256):
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


def plot_sino(sino):
    '''
    Visualise sinogram
    '''
    plt.imshow(sino, cmap='gray')


def TV_SIRT_reg(A, b, Y, n, m, alpha):
    Y = Y.reshape(-1)
    idx1 = np.arange(m, n * m)
    idx2 = np.arange(n * m).reshape(n, m).T[1:].reshape(-1)
    total_variation = alpha * (np.sum(np.abs(Y[idx1] - Y[idx1 - m])) + np.sum(np.abs(Y[idx2] - Y[idx2 - 1])))
    return total_variation


def find_optimal_lambda_SIRT_reg1(b, Y, angles, det_row_count, det_col_count, n, m, alpha):
    '''
    Ищет оптимальный коэффициент в методе наискорейшего спуска, применному к TV-regularization problem
    Использует тернарный поиск
    '''
    l = -100
    r = 100
    grad1 = grad_SIRT1(b, Y, angles, det_row_count, det_col_count, n, m) + alpha * fast_calc_grad(Y, n, m)
    hp, hn, vp, vn = sino_to_lino(b, angles / np.pi * 180, det_col_count)
    hp1, hn1, vp1, vn1 = fht(Y.reshape(n, m))
    hp2, hn2, vp2, vn2 = fht(grad1.reshape(n, m))
    for i in range(40):
        m1 = l + (r - l) / 3
        m2 = r - (r - l) / 3
        val1 = np.linalg.norm(np.concatenate((hp - (hp1 - m1 * hp2), hn - (hn1 - m1 * hn2), 
                                              vp - (vp1 - m1 * vp2), vn - (vn1 - m1 * vn2)))) ** 2
        val2 = np.linalg.norm(np.concatenate((hp - (hp1 - m2 * hp2), hn - (hn1 - m2 * hn2), 
                                              vp - (vp1 - m2 * vp2), vn - (vn1 - m2 * vn2)))) ** 2
        val1 += alpha * TV_SIRT_reg1(Y - m1 * grad1, n, m, alpha)
        val2 += alpha * TV_SIRT_reg1(Y - m2 * grad1, n, m, alpha)
        if val1 < val2:
            r = m2
        else:
            l = m1
    return l


def find_optimal_lambda_SIRT_reg(A, b, Y, n, m, alpha):
    l = -100
    r = 100
    grad = A.T.dot(A.dot(Y) - b) + alpha * fast_calc_grad(Y, n, m)
    #grad = grad_SIRT_reg(backproj_matr, A, b, Y, n, m, alpha)
    for i in range(40):
        m1 = l + (r - l) / 3
        m2 = r - (r - l) / 3
        val1 =  np.linalg.norm(b - A.dot(Y - m1 * grad)) ** 2 + TV_SIRT_reg(A, b, Y - m1 * grad, n, m, alpha)
        val2 =  np.linalg.norm(b - A.dot(Y - m2 * grad)) ** 2 + TV_SIRT_reg(A, b, Y - m2 * grad, n, m, alpha)
        if val1 < val2:
            r = m2
        else:
            l = m1
    return l


def SIRT_reg(A, b, angles=None, det_row_count=None, det_col_count=None, alpha=0, iter=200, threshold=1e-4, show_plots=True):
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
    n = m = b.shape[1]
    start = time.time()
    x = np.zeros((n * m, 1))
    for i in range(iter):
        y = x.copy()
        if A is not None:
            b = b.reshape(-1, 1)
            coef = find_optimal_lambda_SIRT_reg(A, b, x, n, m, alpha)
            x -= coef * (A.T.dot(A.dot(x) - b) + alpha * fast_calc_grad(x, n, m))
        else:
            coef = find_optimal_lambda_SIRT_reg1(b.T, x, angles, det_row_count, det_col_count, n, m, alpha)
            x -= coef * (grad_SIRT1(b.T, x, angles, det_row_count, det_col_count, n, m) + 
                         alpha * fast_calc_grad(x, n, m))
        if np.linalg.norm(y - x) < threshold:
            break
        if i % 10 == 0 and show_plots:
            if A is not None:
                if alpha < 1e-6:
                    show_img(x, i, 'SIRT-sino')
                else:
                    show_img(x, i, 'SIRT-reg-sino')
            else:
                if alpha < 1e-6:
                    show_img(x[::-1, ::-1], i, 'SIRT-breidy')
                else:
                    show_img(x[::-1, ::-1], i, 'SIRT-reg-breidy')
            plt.show()
    end = time.time()
    if A is not None:
        if alpha < 1e-6:
            show_img(x, iter, 'SIRT-sino')
        else:
            show_img(x, iter, 'SIRT-reg-sino')
    else:
        if alpha < 1e-6:
            show_img(x[::-1, ::-1], iter, 'SIRT-breidy')
        else:
            show_img(x[::-1, ::-1], iter, 'SIRT-reg-breidy')
    plt.show()
    print('Время работы SIRT с TV-регуляризацией: {} секунд'.format(end - start))
    return x


def get_A_b(img, angles, det_row_count=1, det_col_count=256):
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
    proj_geom = astra.create_proj_geom('parallel', det_row_count, det_col_count, angles)
    proj_id = astra.create_projector('linear', proj_geom, vol_geom)
    matrix_id = astra.projector.matrix(proj_id)
    A = astra.matrix.get(matrix_id).astype(np.float)
    sinogram_id, sino = astra.create_sino(img, proj_id)
    return A, sino


def prepare_image(img, n, m):
    '''
    Resize image to nxm and gray-scale it
    '''
    img = img.resize((n, m))
    img = np.asarray(img)
    img = rgb2gray(img)
    return img


def SIRT_libr(img, angles, det_row_count=1, det_col_count=256):
    '''
    Стандартный библиотечный SIRT
    '''
    vol_geom = astra.create_vol_geom(img.shape[0], img.shape[1])
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


def phantom_test(angles, det_row_count=1, det_col_count=256, alpha=0.1, path='drive/MyDrive/Диплом/SheppLogan_Phantom.png'):
    '''
    Делает реконструкцию фантома, используя алгоритм Брейди или метод обратной проекции
    params:
    angles - измеренные углы в методе обратной проекции
    alpha - коэффициент для TV-регуляризации
    path - путь к исходной картинке
    det_row_count, det_col_count, angles - параметры детектора
    '''
    img = Image.open(path) #path to image
    img = prepare_image(img, 256, 256)
    A, b = get_A_b(img, angles, det_row_count, det_col_count)
    plt.title('Фантом')
    plt.imshow(img, cmap='gray')
    plt.show()
    plt.title('Синограмма')
    plot_sino(b)
    plt.show()
    SIRT_libr(img, angles)
    SIRT_reg(A, b, alpha=0)
    SIRT_reg(None, b, alpha=0, angles=angles, det_row_count=det_row_count, det_col_count=det_col_count)
    SIRT_reg(A, b, alpha=10)
    SIRT_reg(None, b, alpha=3, angles=angles, det_row_count=det_row_count, det_col_count=det_col_count)


phantom_test(path='Phantom.png', angles=np.linspace(0, np.pi, 18))
