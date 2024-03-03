
#include <pcl/pcl_config.h>
#ifdef HAVE_QHULL

#ifndef PCL_SURFACE_IMPL_CONCAVE_HULL_H_
#define PCL_SURFACE_IMPL_CONCAVE_HULL_H_

#include <map>
#include <pcl/surface/concave_hull.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/io.h>
#include <cstdio>
#include <cstdlib>
#include <pcl/surface/qhull.h>

//////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void pcl::ConcaveHull<PointInT>::reconstruct(PointCloud &output)
{
  output.header = input_->header;
  if (alpha_ <= 0)
  {
    PCL_ERROR("[pcl::%s::reconstruct] Alpha parameter must be set to a positive number!\n", getClassName().c_str());
    output.clear();
    return;
  }

  if (!initCompute())
  {
    output.clear();
    return;
  }

  // Perform the actual surface reconstruction
  std::vector<pcl::Vertices> polygons;
  performReconstruction(output, polygons);

  output.width = output.size();
  output.height = 1;
  output.is_dense = true;

  deinitCompute();
}

//////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void pcl::ConcaveHull<PointInT>::reconstruct(PointCloud &output, std::vector<pcl::Vertices> &polygons)
{
  output.header = input_->header;
  if (alpha_ <= 0)
  {
    PCL_ERROR("[pcl::%s::reconstruct] Alpha parameter must be set to a positive number!\n", getClassName().c_str());
    output.clear();
    return;
  }

  if (!initCompute())
  {
    output.clear();
    return;
  }

  // Perform the actual surface reconstruction
  performReconstruction(output, polygons);

  output.width = output.size();
  output.height = 1;
  output.is_dense = true;

  deinitCompute();
}

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
//////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void pcl::ConcaveHull<PointInT>::performReconstruction(PointCloud &alpha_shape, std::vector<pcl::Vertices> &polygons)
{
  Eigen::Vector4d xyz_centroid;
  // compute centroid of points of the specified index in the cloud
  compute3DCentroid(*input_, *indices_, xyz_centroid);
  EIGEN_ALIGN16 Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Zero();
  // compute cavirance of points of the specified index in the cloud
  computeCovarianceMatrixNormalized(*input_, *indices_, xyz_centroid, covariance_matrix);

  // Check if the covariance matrix is finite or not.
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
      if (!std::isfinite(covariance_matrix.coeffRef(i, j)))
      {
        return;
      }
  }

  EIGEN_ALIGN16 Eigen::Vector3d eigen_values; // ascending
  EIGEN_ALIGN16 Eigen::Matrix3d eigen_vectors;
  pcl::eigen33(covariance_matrix, eigen_vectors, eigen_values);

  Eigen::Affine3d transform1;
  transform1.setIdentity();

  // If no input dimension is specified, determine automatically
  if (dim_ == 0)
  {
    PCL_DEBUG("[pcl::%s] WARNING: Input dimension not specified.  Automatically determining input dimension.\n", getClassName().c_str());
    // min eigen value is very small, this cloud expand in directions of other two eigen vectors. It's plane
    if (std::abs(eigen_values[0]) < std::numeric_limits<double>::epsilon() ||
        std::abs(eigen_values[0] / eigen_values[2]) < 1.0e-3)
    {
      dim_ = 2;
    }
    else
    {
      dim_ = 3;
    }
  }

  if (dim_ == 2)
  {
    // we have points laying on a plane, using 2d convex hull
    // compute transformation bring eigen_vectors.col(i) to z-axis

    transform1(2, 0) = eigen_vectors(0, 0);
    transform1(2, 1) = eigen_vectors(1, 0);
    transform1(2, 2) = eigen_vectors(2, 0);

    transform1(1, 0) = eigen_vectors(0, 1);
    transform1(1, 1) = eigen_vectors(1, 1);
    transform1(1, 2) = eigen_vectors(2, 1);
    transform1(0, 0) = eigen_vectors(0, 2);
    transform1(0, 1) = eigen_vectors(1, 2);
    transform1(0, 2) = eigen_vectors(2, 2);
  }
  else
  {
    transform1.setIdentity();
  }

  PointCloud cloud_transformed;
  // translate cloud to use centroid as new origin of coordinates(points' coordinate minus centroid's coordinate )
  pcl::demeanPointCloud(*input_, *indices_, xyz_centroid, cloud_transformed);
  // if 2d, bing normal of plane to z-axis
  pcl::transformPointCloud(cloud_transformed, cloud_transformed, transform1);

  // True if qhull should free points in qh_freeqhull() or reallocation
  boolT ismalloc = True;

  // option flags for qhull, see qh_opt.htm
  char flags[] = "qhull d QJ";

  // output from qh_produce_output(), use NULL to skip qh_produce_output()
  FILE *outfile = nullptr;

  // error messages from qhull code
  FILE *errfile = stderr;

  // 0 if no error from qhull
  int exitcode;

  // Array of coordinates for each point
  coordT *points = reinterpret_cast<coordT *>(calloc(cloud_transformed.size() * dim_, sizeof(coordT)));

  for (std::size_t i = 0; i < cloud_transformed.size(); ++i)
  {
    points[i * dim_ + 0] = static_cast<coordT>(cloud_transformed[i].x);
    points[i * dim_ + 1] = static_cast<coordT>(cloud_transformed[i].y);

    if (dim_ > 2)
    {
      points[i * dim_ + 2] = static_cast<coordT>(cloud_transformed[i].z);
    }
  }

  /* qhT 结构体封装了运行 Qhull 算法所需的所有全局数据和配置参数，包括输入点、凸包的面和顶点、错误处理机制、内存管理等。
   * 几何数据：输入点集、计算得到的凸包顶点、边和面等。
   * 算法参数：控制算法行为的参数，例如容差设置、选项标志等。
   * 状态信息：记录算法执行过程中的状态信息，比如错误码、统计数据等。
   * 内存管理：用于管理算法执行过程中动态分配的内存。
   */
  qhT qh_qh;
  qhT *qh = &qh_qh;
  QHULL_LIB_CHECK
  qh_zero(qh, errfile);

  // Compute concave hull
  // qh_new_ qhull 函数的主要作用是实现 QuickHull 算法，用于计算一组点在三维空间中的凸包
  /*
  int dim: 表示输入点的维度。例如，对于三维空间中的点，dim 应该设置为 3。
  int numpoints: 表示输入点集中点的总数。
  coordT *points: 是指向坐标点数组的指针，其中每个点由 dim 个坐标组成。
  boolT ismalloc: 指示 points 数组是否是动态分配的。如果是，则在处理完成后，QHull 会负责释放这块内存。如果你的点数组是静态分配的或由其他方式管理，应该传递 False。
  char *qhull_cmd: 是一个字符串，包含了要执行的 QHull 命令。这些命令控制 QHull 的行为，比如生成凸包（"qhull"），构建 Delaunay 三角剖分（"delaunay"），构建 Voronoi 图（"voronoi"）等。此参数还可以包括其他选项，以调整算法的行为，如控制输出的详细程度等。
  FILE *outfile: 用于输出 QHull 的结果。这可以是任何有效的 FILE* 指针，包括 stdout。如果你不需要输出结果，可以传递 NULL。
  FILE *errfile: 用于输出 QHull 的错误信息和日志。同样，这可以是任何有效的 FILE* 指针，包括 stderr。如果你不关心错误信息，可以传递 NULL。
   */
  exitcode = qh_new_qhull(qh, dim_, static_cast<int>(cloud_transformed.size()), points, ismalloc, flags, outfile, errfile);

  if (exitcode != 0)
  {
    PCL_ERROR("[pcl::%s::performReconstrution] ERROR: qhull was unable to compute a "
              "concave hull for the given point cloud (%zu)!\n",
              getClassName().c_str(),
              static_cast<std::size_t>(cloud_transformed.size()));

    // check if it fails because of NaN values...
    if (!cloud_transformed.is_dense)
    {
      bool NaNvalues = false;
      for (std::size_t i = 0; i < cloud_transformed.size(); ++i)
      {
        if (!std::isfinite(cloud_transformed[i].x) ||
            !std::isfinite(cloud_transformed[i].y) ||
            !std::isfinite(cloud_transformed[i].z))
        {
          NaNvalues = true;
          break;
        }
      }

      if (NaNvalues)
      {
        PCL_ERROR("[pcl::%s::performReconstruction] ERROR: point cloud contains NaN values, consider running pcl::PassThrough filter first to remove NaNs!\n",
                  getClassName().c_str());
      }
    }

    alpha_shape.resize(0);
    alpha_shape.width = alpha_shape.height = 0;
    polygons.resize(0);

    qh_freeqhull(qh, !qh_ALL);
    int curlong, totlong;
    qh_memfreeshort(qh, &curlong, &totlong);

    return;
  }

  // 计算并设置 Voronoi 图的顶点
  qh_setvoronoi_all(qh);

  int num_vertices = qh->num_vertices;
  alpha_shape.resize(num_vertices);

  vertexT *vertex; // Voronoi 图的顶点
  // Max vertex id
  int max_vertex_id = 0;
  FORALLvertices
  {
    if (vertex->id + 1 > static_cast<unsigned>(max_vertex_id))
    {
      max_vertex_id = vertex->id + 1;
    }
  }

  /* facetT 是一个核心的结构体，用于表示凸包、Voronoi 图或Delaunay 三角剖分中的一个面（facet）。
     facetT 结构体存储了定义凸包或其他几何结构中一个面的所有必要信息，包括面的顶点、邻接面、以及其他几何和拓扑属性。
   */
  facetT *facet; // set by FORALLfacets

  ++max_vertex_id;
  std::vector<int> qhid_to_pcidx(max_vertex_id);

  int num_facets = qh->num_facets;

  if (dim_ == 3)
  {
    setT *triangles_set = qh_settemp(qh, 4 * num_facets);
    if (voronoi_centers_)
    {
      voronoi_centers_->points.resize(num_facets);
    }

    int non_upper = 0;
    FORALLfacets
    {
      // Facets are tetrahedrons (3d)
      if (!facet->upperdelaunay)
      {
        auto *anyVertex = static_cast<vertexT *>(facet->vertices->e[0].p);
        double *center = facet->center;
        double r = qh_pointdist(anyVertex->point, center, dim_);

        if (voronoi_centers_)
        {
          (*voronoi_centers_)[non_upper].x = static_cast<float>(facet->center[0]);
          (*voronoi_centers_)[non_upper].y = static_cast<float>(facet->center[1]);
          (*voronoi_centers_)[non_upper].z = static_cast<float>(facet->center[2]);
        }

        non_upper++;

        if (r <= alpha_)
        {
          // all triangles in tetrahedron are good, add them all to the alpha shape (triangles_set)
          qh_makeridges(qh, facet);
          facet->good = true;
          facet->visitid = qh->visit_id;
          ridgeT *ridge, **ridgep;
          FOREACHridge_(facet->ridges)
          {
            facetT *neighb = otherfacet_(ridge, facet);
            if ((neighb->visitid != qh->visit_id))
            {
              qh_setappend(qh, &triangles_set, ridge);
            }
          }
        }
        else
        {
          // consider individual triangles from the tetrahedron...
          facet->good = false;
          facet->visitid = qh->visit_id;
          qh_makeridges(qh, facet);
          ridgeT *ridge, **ridgep;
          FOREACHridge_(facet->ridges)
          {
            facetT *neighb;
            neighb = otherfacet_(ridge, facet);
            if ((neighb->visitid != qh->visit_id))
            {
              // check if individual triangle is good and add it to triangles_set

              PointInT a, b, c;
              a.x = static_cast<float>((static_cast<vertexT *>(ridge->vertices->e[0].p))->point[0]);
              a.y = static_cast<float>((static_cast<vertexT *>(ridge->vertices->e[0].p))->point[1]);
              a.z = static_cast<float>((static_cast<vertexT *>(ridge->vertices->e[0].p))->point[2]);
              b.x = static_cast<float>((static_cast<vertexT *>(ridge->vertices->e[1].p))->point[0]);
              b.y = static_cast<float>((static_cast<vertexT *>(ridge->vertices->e[1].p))->point[1]);
              b.z = static_cast<float>((static_cast<vertexT *>(ridge->vertices->e[1].p))->point[2]);
              c.x = static_cast<float>((static_cast<vertexT *>(ridge->vertices->e[2].p))->point[0]);
              c.y = static_cast<float>((static_cast<vertexT *>(ridge->vertices->e[2].p))->point[1]);
              c.z = static_cast<float>((static_cast<vertexT *>(ridge->vertices->e[2].p))->point[2]);

              double r = pcl::getCircumcircleRadius(a, b, c);
              if (r <= alpha_)
              {
                qh_setappend(qh, &triangles_set, ridge);
              }
            }
          }
        }
      }
    }

    if (voronoi_centers_)
    {
      voronoi_centers_->points.resize(non_upper);
    }

    // filter, add points to alpha_shape and create polygon structure

    int num_good_triangles = 0;
    ridgeT *ridge, **ridgep;
    FOREACHridge_(triangles_set)
    {
      if (ridge->bottom->upperdelaunay || ridge->top->upperdelaunay || !ridge->top->good || !ridge->bottom->good)
      {
        num_good_triangles++;
      }
    }

    polygons.resize(num_good_triangles);

    int vertices = 0;
    std::vector<bool> added_vertices(max_vertex_id, false);

    int triangles = 0;
    FOREACHridge_(triangles_set)
    {
      if (ridge->bottom->upperdelaunay || ridge->top->upperdelaunay || !ridge->top->good || !ridge->bottom->good)
      {
        polygons[triangles].vertices.resize(3);
        int vertex_n, vertex_i;
        FOREACHvertex_i_(qh, (*ridge).vertices) // 3 vertices per ridge!
        {
          if (!added_vertices[vertex->id])
          {
            alpha_shape[vertices].x = static_cast<float>(vertex->point[0]);
            alpha_shape[vertices].y = static_cast<float>(vertex->point[1]);
            alpha_shape[vertices].z = static_cast<float>(vertex->point[2]);

            qhid_to_pcidx[vertex->id] = vertices; // map the vertex id of qhull to the point cloud index
            added_vertices[vertex->id] = true;
            vertices++;
          }

          polygons[triangles].vertices[vertex_i] = qhid_to_pcidx[vertex->id];
        }

        triangles++;
      }
    }

    alpha_shape.resize(vertices);
    alpha_shape.width = alpha_shape.size();
    alpha_shape.height = 1;
  } // end if dim_=3
  else
  {
    // Compute the alpha complex for the set of points
    // Filters the delaunay triangles
    /* setT 是一个用于表示通用集合的结构体，它提供了一种灵活的方式来管理和存储数据集合，例如点、面（facetT）、顶点（vertexT）等。
       setT 结构体设计用于支持 Qhull 中的多种数据管理需求，特别是在处理凸包、Voronoi 图、Delaunay 三角剖分等几何结构时，
       对于存储和操作这些结构的组成部分（如顶点集合、面集合）非常有用。

       qh_settemp用于创建一个指定大小的临时 setT 集合，之后可以将需要临时管理的数据添加到这个集合中。
       完成临时数据处理后，应使用相应的函数（如 qh_settempfree）来释放这个临时集合，以确保不会发生内存泄露。
    */
    setT *edges_set = qh_settemp(qh, 3 * num_facets); // every face consists of three edges
    // if need voronoi, should call setVoronoiCenters to set pointer first
    if (voronoi_centers_)
    {
      voronoi_centers_->points.resize(num_facets);
    }

    int dd = 0; //
    FORALLfacets
    {
      // Facets are the delaunay triangles (2d)
      if (!facet->upperdelaunay)
      {
        // Check if the distance from any vertex to the facet->center
        // (center of the voronoi cell) is smaller than alpha
        auto *anyVertex = static_cast<vertexT *>(facet->vertices->e[0].p);
        double r = (sqrt((anyVertex->point[0] - facet->center[0]) * (anyVertex->point[0] - facet->center[0]) +
                         (anyVertex->point[1] - facet->center[1]) * (anyVertex->point[1] - facet->center[1])));
        if (r <= alpha_)
        {
          pcl::Vertices facet_vertices; // TODO: is not used!!

          /* qh_makeridges通过遍历凸包中的所有面，然后对于每个面，查找与之相邻的面。对于每一对相邻的面，函数确定它们的共享边界，
             并创建相应的脊对象。
           */
          qh_makeridges(qh, facet);
          facet->good = true;

          ridgeT *ridge, **ridgep;
          // FOREACHridge_ 用于遍历所有脊（ridgeT 类型）的宏。
          // 把一个面的所有脊添加到edge集合中
          FOREACHridge_(facet->ridges) // 用于遍历一个面（facetT 类型）的所有脊（ridgeT 类型）
          {
            /* qh_setappend 函数用于向一个 setT 类型的集合中添加一个新元素
             */
            qh_setappend(qh, &edges_set, ridge);
          }

          if (voronoi_centers_)
          {
            (*voronoi_centers_)[dd].x = static_cast<float>(facet->center[0]);
            (*voronoi_centers_)[dd].y = static_cast<float>(facet->center[1]);
            (*voronoi_centers_)[dd].z = 0.0f;
          }

          ++dd;
        }
        else
        {
          facet->good = false;
        }
      }
    } // endfor: add ridges of all face

    int vertices = 0;
    std::vector<bool> added_vertices(max_vertex_id, false);
    std::map<int, std::vector<int>> edges;

    ridgeT *ridge, **ridgep;
    FOREACHridge_(edges_set)
    {
      if (ridge->bottom->upperdelaunay || ridge->top->upperdelaunay || !ridge->top->good || !ridge->bottom->good)
      {
        int vertex_n, vertex_i;
        int vertices_in_ridge = 0;
        std::vector<int> pcd_indices;
        pcd_indices.resize(2);

        FOREACHvertex_i_(qh, (*ridge).vertices) // in 2-dim, 2 vertices per ridge!
        {
          if (!added_vertices[vertex->id])
          {
            alpha_shape[vertices].x = static_cast<float>(vertex->point[0]);
            alpha_shape[vertices].y = static_cast<float>(vertex->point[1]);

            if (dim_ > 2)
            {
              alpha_shape[vertices].z = static_cast<float>(vertex->point[2]);
            }
            else
            {
              alpha_shape[vertices].z = 0;
            }

            qhid_to_pcidx[vertex->id] = vertices; // map the vertex id of qhull to the point cloud index
            added_vertices[vertex->id] = true;
            pcd_indices[vertices_in_ridge] = vertices; //
            vertices++;
          }
          else
          {
            pcd_indices[vertices_in_ridge] = qhid_to_pcidx[vertex->id];
          }

          vertices_in_ridge++;
        } // endfor: add all vertices of current ridge

        // make edges bidirectional and pointing to alpha_shape pointcloud...
        edges[pcd_indices[0]].push_back(pcd_indices[1]);
        edges[pcd_indices[1]].push_back(pcd_indices[0]);
      } // endif

    } // endfor: edges(ridges)

    alpha_shape.resize(vertices);

    PointCloud alpha_shape_sorted;
    alpha_shape_sorted.resize(vertices);

    // iterate over edges until they are empty!
    auto curr = edges.begin();
    int next = -1;
    std::vector<bool> used(vertices, false); // used to decide which direction should we take!
    std::vector<int> pcd_idx_start_polygons;
    pcd_idx_start_polygons.push_back(0);

    // start following edges and removing elements
    int sorted_idx = 0;
    while (!edges.empty())
    {
      alpha_shape_sorted[sorted_idx] = alpha_shape[(*curr).first];
      // check where we can go from (*curr).first
      for (const auto &i : (*curr).second)
      {
        if (!used[i])
        {
          // we can go there
          next = i;
          break;
        }
      }

      used[(*curr).first] = true;
      edges.erase(curr); // remove edges starting from curr

      sorted_idx++;

      if (edges.empty())
        break;

      // reassign current
      curr = edges.find(next); // if next is not found, then we have unconnected polygons.
      if (curr == edges.end())
      {
        // set current to any of the remaining in edge!
        curr = edges.begin();
        pcd_idx_start_polygons.push_back(sorted_idx);
      }
    } // endwhile: store all vertices

    pcd_idx_start_polygons.push_back(sorted_idx);

    alpha_shape.points = alpha_shape_sorted.points;

    polygons.reserve(pcd_idx_start_polygons.size() - 1);

    for (std::size_t poly_id = 0; poly_id < pcd_idx_start_polygons.size() - 1; poly_id++)
    {
      // Check if we actually have a polygon, and not some degenerated output from QHull
      if (pcd_idx_start_polygons[poly_id + 1] - pcd_idx_start_polygons[poly_id] >= 3)
      {
        pcl::Vertices vertices;
        vertices.vertices.resize(pcd_idx_start_polygons[poly_id + 1] - pcd_idx_start_polygons[poly_id]);
        // populate points in the corresponding polygon
        for (int j = pcd_idx_start_polygons[poly_id]; j < pcd_idx_start_polygons[poly_id + 1]; ++j)
        {
          vertices.vertices[j - pcd_idx_start_polygons[poly_id]] = static_cast<std::uint32_t>(j);
        } // endfor: current polygons

        polygons.push_back(vertices);
      }
    } // endfor: all polygons

    if (voronoi_centers_)
    {
      voronoi_centers_->points.resize(dd);
    }
  } // endelse dim_=2

  qh_freeqhull(qh, !qh_ALL);
  int curlong, totlong;
  qh_memfreeshort(qh, &curlong, &totlong);

  Eigen::Affine3d transInverse = transform1.inverse();
  pcl::transformPointCloud(alpha_shape, alpha_shape, transInverse);
  xyz_centroid[0] = -xyz_centroid[0];
  xyz_centroid[1] = -xyz_centroid[1];
  xyz_centroid[2] = -xyz_centroid[2];
  pcl::demeanPointCloud(alpha_shape, xyz_centroid, alpha_shape);

  // also transform voronoi_centers_...
  if (voronoi_centers_)
  {
    pcl::transformPointCloud(*voronoi_centers_, *voronoi_centers_, transInverse);
    pcl::demeanPointCloud(*voronoi_centers_, xyz_centroid, *voronoi_centers_);
  }

  if (keep_information_)
  {
    // build a tree with the original points
    pcl::KdTreeFLANN<PointInT> tree(true);
    tree.setInputCloud(input_, indices_);

    pcl::Indices neighbor;
    std::vector<float> distances;
    neighbor.resize(1);
    distances.resize(1);

    // for each point in the concave hull, search for the nearest neighbor in the original point cloud
    hull_indices_.header = input_->header;
    hull_indices_.indices.clear();
    hull_indices_.indices.reserve(alpha_shape.size());

    for (const auto &point : alpha_shape)
    {
      tree.nearestKSearch(point, 1, neighbor, distances);
      hull_indices_.indices.push_back(neighbor[0]);
    }

    // replace point with the closest neighbor in the original point cloud
    pcl::copyPointCloud(*input_, hull_indices_.indices, alpha_shape);
  }
}
#ifdef __GNUC__
#pragma GCC diagnostic warning "-Wold-style-cast"
#endif

//////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void pcl::ConcaveHull<PointInT>::performReconstruction(PolygonMesh &output)
{
  // Perform reconstruction
  pcl::PointCloud<PointInT> hull_points;
  performReconstruction(hull_points, output.polygons);

  // Convert the PointCloud into a PCLPointCloud2
  pcl::toPCLPointCloud2(hull_points, output.cloud);
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void pcl::ConcaveHull<PointInT>::performReconstruction(std::vector<pcl::Vertices> &polygons)
{
  pcl::PointCloud<PointInT> hull_points;
  performReconstruction(hull_points, polygons);
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT>
void pcl::ConcaveHull<PointInT>::getHullPointIndices(pcl::PointIndices &hull_point_indices) const
{
  hull_point_indices = hull_indices_;
}

#define PCL_INSTANTIATE_ConcaveHull(T) template class PCL_EXPORTS pcl::ConcaveHull<T>;

#endif // PCL_SURFACE_IMPL_CONCAVE_HULL_H_
#endif
