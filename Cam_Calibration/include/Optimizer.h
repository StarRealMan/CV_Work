#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

class VertexHomo : public g2o::BaseVertex<8, Eigen::Matrix3f> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override
    {
        Eigen::Matrix3f init_Homo = Eigen::Matrix3f::Zero();
        init_Homo(2, 2) = 1.0f;
        _estimate = init_Homo;
    }
    
    virtual void oplusImpl(const double *update) override
    {
        Eigen::Matrix3f update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], \
                        update[4], update[5], update[6], update[7], 0.0f;
        _estimate = update_eigen + _estimate;
    }

    virtual bool read(std::istream &in) override
    {
        return true;
    }

    virtual bool write(std::ostream &out) const override
    {
        return true;
    }
};

class EdgeHomo : public g2o::BaseUnaryEdge<2, Eigen::Vector2f, VertexHomo>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeHomo(const Eigen::Vector3f& pos)
    {
        _pos = pos;
    }

    virtual void computeError() override
    {
        const VertexHomo *v = static_cast<VertexHomo *> (_vertices[0]);
        Eigen::Matrix3f T = v->estimate();
        
        Eigen::Vector3f posHomo = T * _pos;
        posHomo = posHomo / posHomo(2);

        _error = _measurement.cast<double>() - posHomo.head<2>().cast<double>();
    }

    virtual void linearizeOplus() override
    {
        const VertexHomo *v = static_cast<VertexHomo *> (_vertices[0]);
        Eigen::Matrix3f T = v->estimate();

        // TODO
        _jacobianOplusXi << ;
    }

    virtual bool read(std::istream &in) override
    {
        return true;
    }

    virtual bool write(std::ostream &out) const override
    {
        return true;
    }

private:
    Eigen::Vector3f _pos;
};


class Optimizer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Optimizer> Ptr;

    Optimizer(Eigen::Matrix3f init_Homo);
    ~Optimizer();

    void doOptimization(uint8_t optim_round);
    void addMeasure(Eigen::Vector3f& world_pt, Eigen::Vector2f& cam_pt, cv::Point2f conf);
    Eigen::Matrix3f GetHomo();

    static uint _edge_id;

private:
    g2o::SparseOptimizer _optimizer;
    VertexHomo *_vertex_Homo;
    Eigen::Matrix3f _optimize_val;
};

#endif

