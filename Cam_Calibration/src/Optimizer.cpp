#include "Optimizer.h"

Optimizer::Optimizer(Eigen::Matrix3f init_Homo)
{
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg
                     (g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    _optimizer.setAlgorithm(solver);

    _vertex_Homo = new VertexHomo();
    _optimize_val = init_Homo;
    _vertex_Homo->setEstimate(_optimize_val);
    _vertex_Homo->setId(0);
    _optimizer.addVertex(_vertex_Homo);
}

Optimizer::~Optimizer()
{

}

void Optimizer::doOptimization(uint8_t optim_round)
{
    _optimizer.setVerbose(true);
    _optimizer.initializeOptimization();
    _optimizer.optimize(optim_round);
}

void Optimizer::addMeasure(Eigen::Vector3f& world_pt, Eigen::Vector2f& cam_pt, cv::Point2f conf)
{
    EdgeHomo *edge = new EdgeHomo(world_pt);
    edge->setId(_edge_id);
    edge->setVertex(0, _vertex_Homo);
    edge->setMeasurement(cam_pt);

    Eigen::Matrix2d Information;
    Information << 1.0, 0.0, 0.0, 1.0;

    edge->setInformation(Information);
    edge->setRobustKernel(new g2o::RobustKernelHuber);
    _optimizer.addEdge(edge);
    _edge_id++;
}

Eigen::Matrix3f Optimizer::GetHomo()
{
    return _vertex_Homo->estimate();
}
