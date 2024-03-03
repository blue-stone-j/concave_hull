
#ifndef PCL_SURFACE_RECONSTRUCTION_IMPL_H_
#define PCL_SURFACE_RECONSTRUCTION_IMPL_H_

#include <pcl/conversions.h>      // for pcl::toPCLPointCloud2
#include <pcl/search/kdtree.h>    // for KdTree
#include <pcl/search/organized.h> // for OrganizedNeighbor

namespace pcl
{

  template <typename PointInT>
  void
  SurfaceReconstruction<PointInT>::reconstruct(pcl::PolygonMesh &output)
  {
    // Copy the header
    output.header = input_->header;

    if (!initCompute())
    {
      output.cloud.width = output.cloud.height = 0;
      output.cloud.data.clear();
      output.polygons.clear();
      return;
    }

    // Check if a space search locator was given
    if (check_tree_)
    {
      if (!tree_)
      {
        if (input_->isOrganized())
          tree_.reset(new pcl::search::OrganizedNeighbor<PointInT>());
        else
          tree_.reset(new pcl::search::KdTree<PointInT>(false));
      }

      // Send the surface dataset to the spatial locator
      tree_->setInputCloud(input_, indices_);
    }

    // Set up the output dataset
    pcl::toPCLPointCloud2(*input_, output.cloud); /// NOTE: passing in boost shared pointer with * as const& should be OK here
    output.polygons.clear();
    output.polygons.reserve(2 * indices_->size()); /// NOTE: usually the number of triangles is around twice the number of vertices
    // Perform the actual surface reconstruction
    performReconstruction(output);

    deinitCompute();
  }

  template <typename PointInT>
  void
  SurfaceReconstruction<PointInT>::reconstruct(pcl::PointCloud<PointInT> &points,
                                               std::vector<pcl::Vertices> &polygons)
  {
    // Copy the header
    points.header = input_->header;

    if (!initCompute())
    {
      points.width = points.height = 0;
      points.clear();
      polygons.clear();
      return;
    }

    // Check if a space search locator was given
    if (check_tree_)
    {
      if (!tree_)
      {
        if (input_->isOrganized())
          tree_.reset(new pcl::search::OrganizedNeighbor<PointInT>());
        else
          tree_.reset(new pcl::search::KdTree<PointInT>(false));
      }

      // Send the surface dataset to the spatial locator
      tree_->setInputCloud(input_, indices_);
    }

    // Set up the output dataset
    polygons.clear();
    polygons.reserve(2 * indices_->size()); /// NOTE: usually the number of triangles is around twice the number of vertices
    // Perform the actual surface reconstruction
    performReconstruction(points, polygons);

    deinitCompute();
  }

  template <typename PointInT>
  void
  MeshConstruction<PointInT>::reconstruct(pcl::PolygonMesh &output)
  {
    // Copy the header
    output.header = input_->header;

    if (!initCompute())
    {
      output.cloud.width = output.cloud.height = 1;
      output.cloud.data.clear();
      output.polygons.clear();
      return;
    }

    // Check if a space search locator was given
    if (check_tree_)
    {
      if (!tree_)
      {
        if (input_->isOrganized())
          tree_.reset(new pcl::search::OrganizedNeighbor<PointInT>());
        else
          tree_.reset(new pcl::search::KdTree<PointInT>(false));
      }

      // Send the surface dataset to the spatial locator
      tree_->setInputCloud(input_, indices_);
    }

    // Set up the output dataset
    pcl::toPCLPointCloud2(*input_, output.cloud); /// NOTE: passing in boost shared pointer with * as const& should be OK here
    //  output.polygons.clear ();
    //  output.polygons.reserve (2*indices_->size ()); /// NOTE: usually the number of triangles is around twice the number of vertices
    // Perform the actual surface reconstruction
    performReconstruction(output);

    deinitCompute();
  }

  template <typename PointInT>
  void
  MeshConstruction<PointInT>::reconstruct(std::vector<pcl::Vertices> &polygons)
  {
    if (!initCompute())
    {
      polygons.clear();
      return;
    }

    // Check if a space search locator was given
    if (check_tree_)
    {
      if (!tree_)
      {
        if (input_->isOrganized())
          tree_.reset(new pcl::search::OrganizedNeighbor<PointInT>());
        else
          tree_.reset(new pcl::search::KdTree<PointInT>(false));
      }

      // Send the surface dataset to the spatial locator
      tree_->setInputCloud(input_, indices_);
    }

    // Set up the output dataset
    // polygons.clear ();
    // polygons.reserve (2 * indices_->size ()); /// NOTE: usually the number of triangles is around twice the number of vertices
    // Perform the actual surface reconstruction
    performReconstruction(polygons);

    deinitCompute();
  }

} // namespace pcl

#endif // PCL_SURFACE_RECONSTRUCTION_IMPL_H_
