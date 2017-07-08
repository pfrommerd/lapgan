import numpy as np
import scipy.spatial

from matplotlib.path import Path

def vorify_batch(image_batch, cellSize, sigma):
    return np.array([vorify(im, cellSize, sigma) for im in image_batch])

def vorify(image, cellSize, sigma):
    # Generate a grid of points
    x, y = np.mgrid[cellSize[0] // 2:image.shape[0] - cellSize[0] // 2:cellSize[0],
                    cellSize[1] // 2:image.shape[1] - cellSize[1] // 2:cellSize[1]]
    x = np.expand_dims(x.ravel(), axis=1)
    y = np.expand_dims(y.ravel(), axis=1)
    points = np.concatenate([x, y], axis=1)
    # Add random noise with variance sigma^2
    noise = np.random.normal(scale=sigma, size=points.shape)

    points = points + noise
    
    vor = scipy.spatial.Voronoi(points)

    regions, vert = voronoi_finite_polygons_2d(vor)

    result = np.zeros(image.shape)
    for region in regions:
        polygon = vert[region]
        mask = np.expand_dims(rasterize_mask(polygon, (image.shape[0], image.shape[1])), axis=2)
        num_pixels = np.sum(mask)
        if num_pixels > 0:
            pixel_sum = np.sum(image * mask, axis=(0,1)) / num_pixels
            mask = np.repeat(mask, 3, axis=2)
            result += mask * pixel_sum

    return result

def rasterize_mask(polygon, shape):
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T

    path = Path(polygon)
    grid = path.contains_points(points)
    grid = grid.reshape(shape)

    return grid.astype(int)

def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from keras.datasets import cifar10
    (xtrain, ytrain_cat), (xtest, ytest_cat) = cifar10.load_data()
    # Process the data
    xtrain = xtrain.astype(np.float32) / 255.0
    xtest = xtest.astype(np.float32) / 255.0
    
    result = vorify(xtest[0], [2,2], 1)

    plt.imshow(result)
    plt.show()
