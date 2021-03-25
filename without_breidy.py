
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image
import astra
import scipy.sparse
import scipy.sparse.linalg
import time

%matplotlib inline


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
    #print(grad.shape)
    return grad.reshape(-1, 1)


def fast_calc_grad_mu(Y, mu=0.1, n=256, m=256):
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
    Visulise sinogram
    '''
    plt.imshow(sino, cmap='gray')


def opt_SIRT(A, b, Y, n, m):
    return np.linalg.norm(b - A.dot(Y))


def opt_SIRT_reg(A, b, Y, n, m, alpha):
    syst = np.linalg.norm(b - A.dot(Y)) ** 2
    Y = Y.reshape(-1)
    idx1 = np.arange(m, n * m)
    idx2 = np.arange(n * m).reshape(n, m).T[1:].reshape(-1)
    total_variation = alpha * (np.sum(np.abs(Y[idx1] - Y[idx1 - m])) + np.sum(np.abs(Y[idx2] - Y[idx2 - 1])))
    return syst + total_variation


def opt_SIRT_reg_mu(A, b, Y, n, m, alpha, mu):
    syst = np.linalg.norm(b - A.dot(Y))
    Y = Y.reshape(-1)
    idx1 = np.arange(m, n * m)
    idx2 = np.arange(n * m).reshape(n, m).T[1:].reshape(-1)
    total_variation_mu = alpha * (np.sum(((Y[idx1] - Y[idx1 - m]) ** 2 + mu) ** 0.5) + np.sum(((Y[idx2] - Y[idx2 - 1]) ** 2 + mu) ** 0.5))
    return syst + total_variation_mu


def grad_SIRT(backproj_matr, A, b, Y, n, m):
    return -backproj_matr.dot(b - A.dot(Y))


def grad_SIRT_reg(backproj_matr, A, b, Y, n, m, alpha):
    return backproj_matr.dot(A.dot(Y) - b) + alpha * fast_calc_grad(Y, n, m)


def grad_SIRT_reg_mu(backproj_matr, A, b, Y, n, m, alpha, mu):
    return backproj_matr.dot(A.dot(Y) - b) + alpha * fast_calc_grad_mu(Y, mu, n, m)


def find_optimal_lambda_SIRT(backproj_matr, A, b, Y, n, m):
    l = -1000
    r = 1000
    for i in range(40):
        m1 = l + (r - l) / 3
        m2 = r - (r - l) / 3
        grad1 = grad_SIRT(backproj_matr, A, b, Y, n, m)
        val1 = opt_SIRT(A, b, Y - m1 * grad1, n, m)
        grad2 = grad_SIRT(backproj_matr, A, b, Y, n, m)
        val2 = opt_SIRT(A, b, Y - m2 * grad2, n, m)
        if val1 < val2:
            r = m2
        else:
            l = m1
    return l


def find_optimal_lambda_SIRT_reg(backproj_matr, A, b, Y, n, m, alpha):
    l = -1000
    r = 1000
    for i in range(40):
        m1 = l + (r - l) / 3
        m2 = r - (r - l) / 3
        grad1 = grad_SIRT_reg(backproj_matr, A, b, Y, n, m, alpha)
        val1 = opt_SIRT_reg(A, b, Y - m1 * grad1, n, m, alpha)
        grad2 = grad_SIRT_reg(backproj_matr, A, b, Y, n, m, alpha)
        val2 = opt_SIRT_reg(A, b, Y - m2 * grad2, n, m, alpha)
        if val1 < val2:
            r = m2
        else:
            l = m1
    return l


def find_optimal_lambda_SIRT_reg_mu(backproj_matr, A, b, Y, n, m, alpha, mu):
    l = -1000
    r = 1000
    for i in range(40):
        m1 = l + (r - l) / 3
        m2 = r - (r - l) / 3
        grad1 = grad_SIRT_reg_mu(backproj_matr, A, b, Y, n, m, alpha, mu)
        val1 = opt_SIRT_reg_mu(A, b, Y - m1 * grad1, n, m, alpha, mu)
        grad2 = grad_SIRT_reg_mu(backproj_matr, A, b, Y, n, m, alpha, mu)
        val2 = opt_SIRT_reg_mu(A, b, Y - m2 * grad2, n, m, alpha, mu)
        if val1 < val2:
            r = m2
        else:
            l = m1
    return l


def SIRT(A, b, n=256, m=256, iter=200, threshold=1e-4, show_plots=True):
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
    C1 = np.array(1 / scipy.sparse.csr_matrix.sum(A, axis=0)).reshape(-1)
    R1 = np.array(1 / scipy.sparse.csr_matrix.sum(A, axis=1)).reshape(-1)
    elem = [np.arange(C1.shape[0]), np.arange(C1.shape[0])]
    elem1 = [range(R1.shape[0]), range(R1.shape[0])]
    C = scipy.sparse.csr_matrix((C1, (elem[0], elem[1])), shape=(len(C1), len(C1)))
    R = scipy.sparse.csr_matrix((R1, (elem1[0], elem1[1])), shape=(len(R1), len(R1)))
    b = b.reshape(-1, 1)
    x = np.zeros((n * m, 1))
    #backproj_matr = (C.dot(A.T)).dot(R)
    backproj_matr = A.T
    #x = np.zeros((256 * 256))
    for i in range(iter):
        y = x
        #backproj_matr = (C.dot(A.T)).dot(R)
        x = x + find_optimal_lambda_SIRT(backproj_matr, A, b, x, n, m) * backproj_matr.dot(b - A.dot(x))
        if np.linalg.norm(y - x) < threshold:
            break
        if i % 10 == 0 and show_plots:
            show_img(x, i, 'SIRT')
            plt.show()
    end = time.time()
    show_img(x, iter, 'SIRT')
    plt.show()
    print('Время работы SIRT: {} секунд'.format(end - start))
    return x


def SIRT_reg(A, b, n=256, m=256, alpha=0.02, iter=200, threshold=1e-4, show_plots=True):
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
    C1 = np.array(1 / scipy.sparse.csr_matrix.sum(A, axis=0)).reshape(-1)
    R1 = np.array(1 / scipy.sparse.csr_matrix.sum(A, axis=1)).reshape(-1)
    elem = [np.arange(C1.shape[0]), np.arange(C1.shape[0])]
    elem1 = [range(R1.shape[0]), range(R1.shape[0])]
    C = scipy.sparse.csr_matrix((C1, (elem[0], elem[1])), shape=(len(C1), len(C1)))
    R = scipy.sparse.csr_matrix((R1, (elem1[0], elem1[1])), shape=(len(R1), len(R1)))
    b = b.reshape(-1, 1)
    x = np.zeros((n * m, 1))
    #backproj_matr = (C.dot(A.T)).dot(R)
    backproj_matr = A.T
    #x = np.zeros((256 * 256))
        #x += scipy.sparse.csc_matrix.dot(scipy.sparse.csc_matrix.dot(scipy.sparse.csc_matrix.dot(C, A.T), R), b - scipy.sparse.csc_matrix.dot(A, scipy.sparse.csc_matrix(x)))
    #x += -alpha * calc_grad_mu(x, 0.1)
    for i in range(iter):
        #backproj_matr = (C.dot(A.T)).dot(R)
        y = x.copy()
        coef = find_optimal_lambda_SIRT_reg(backproj_matr, A, b, x, n, m, alpha)
        #x = x + coef * backproj_matr.dot(b - A.dot(x)) + coef * -alpha * fast_calc_grad(x)
        x -= (coef) * grad_SIRT_reg(backproj_matr, A, b, x, n, m, alpha)
        #x += coef * -alpha * fast_calc_grad(x)
        if np.linalg.norm(y - x) < threshold:
            break
        if i % 1 == 0 and show_plots:
            show_img(x, i, 'SIRT-reg')
            plt.show()
    end = time.time()
    show_img(x, iter, 'SIRT-reg')
    plt.show()
    print('Время работы SIRT с TV-регуляризацией: {} секунд'.format(end - start))
    return x


def SIRT_reg_mu(A, b, n=256, m=256, alpha=0.01, mu=0.01, iter=200, threshold=1e-4, show_plots=False):
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
        if i % 1 == 0 and show_plots:
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


def prepare_image(img, n, m):
    '''
    Resize image to nxm and gray-scale it
    '''
    img = img.resize((n, m))
    img = np.asarray(img)
    img = rgb2gray(img)
    return img


def SIRT_libr(img, angles=18, det_row_count=1, det_col_count=256):
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


def phantom_test(angles=18, det_row_count=1, det_col_count=256, alpha=40, mu=0.1, path='drive/MyDrive/Диплом/SheppLogan_Phantom.png'):
    img = Image.open(path) #path to image
    img = prepare_image(img, 256, 256)
    print(img.min(), img.max())
    A, b = get_A_b(img, angles, det_row_count, det_col_count)
    plt.title('Фантом')
    plt.imshow(img, cmap='gray')
    plt.show()
    plt.title('Синограмма')
    plot_sino(b)
    plt.show()
    #SIRT_libr(img)
    #SIRT(A, b)
    SIRT_reg(A, b, alpha=alpha)
    #SIRT_reg_mu(A, b)

phantom_test(path="Phantom.png")
