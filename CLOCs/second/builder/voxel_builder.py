import numpy as np

try:
    from spconv.utils import VoxelGenerator
except ImportError:
    VoxelGenerator = None
    import cumm.tensorview as tv
    from spconv.utils import Point2VoxelCPU3d
from second.protos import voxel_generator_pb2


class VoxelGeneratorCompat:
    """spconv v2 wrapper that preserves the old SECOND VoxelGenerator API."""

    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
        grid = (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size
        self.grid_size = np.round(grid).astype(np.int64)
        self._generator = Point2VoxelCPU3d(
            voxel_size,
            point_cloud_range,
            4,
            max_voxels,
            max_num_points,
        )

    def generate(self, points, max_voxels=None):
        voxels, coordinates, num_points = self._generator.point_to_voxel(tv.from_numpy(points.astype(np.float32)))
        return voxels.numpy(), coordinates.numpy(), num_points.numpy()


def build(voxel_config):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(voxel_config, (voxel_generator_pb2.VoxelGenerator)):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')
    generator_cls = VoxelGenerator or VoxelGeneratorCompat
    voxel_generator = generator_cls(
        voxel_size=list(voxel_config.voxel_size),
        point_cloud_range=list(voxel_config.point_cloud_range),
        max_num_points=voxel_config.max_number_of_points_per_voxel,
        max_voxels=20000)
    return voxel_generator
