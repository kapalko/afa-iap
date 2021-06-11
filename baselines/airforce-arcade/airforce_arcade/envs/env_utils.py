import numpy as np


def process_raycast(raycast, obs_shape, reorder_mat):
    raycast = raycast[1::2] # remove hit flags
    raycast = np.reshape(raycast, obs_shape)
    raycast = np.matmul(raycast, reorder_mat)
    raycast = 255 - (raycast * 255.0).astype(np.uint8)
    return raycast


def reorder(arr):
    arr_len = len(arr)
    newarr = np.zeros(arr_len)
    middle = int((arr_len-1)/2)
    newarr[middle] = arr[0]
    j = 1
    for i in range(1,arr_len-middle):
        newarr[middle + i] = arr[j]
        newarr[middle - i] = arr[j+1]
        j += 2
    return newarr


def generate_reorder_mat(totalRays):
    raysPerSide = int(np.floor(totalRays/2))
    reorderMatrix = np.zeros([totalRays,totalRays])
    # fill in first half of the matrix
    for (row,col) in zip(np.array(range(totalRays-1,0,-2)),np.array(range(0,raysPerSide))):
        reorderMatrix[row,col] = 1
    # fill in middle value
    reorderMatrix[0,raysPerSide] = 1
    # fill in second half of matrix
    for (row,col) in zip(np.array(range(1,totalRays-1,2)),np.array(range(raysPerSide+1,2*raysPerSide+1))):
        reorderMatrix[row,col] = 1
    return reorderMatrix
