#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>	//	pcl::transformPointCloud 用到这个头文件
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl/point_types.h>

#include <json/json.h>
#include <json/value.h>
#include <fstream>
#include <algorithm>

#include <vector>
#include <string>

#include "dbscan.h"

using namespace Eigen;
using namespace std;

#define MAX_DIS 10//2.3
# define M_PI_2		1.57079632679489661923

// created for ceres
struct Pl_ICP 
    {
        Pl_ICP(Eigen::Vector3d cur_point_, Eigen::Vector3d last_point_a_, Eigen::Vector3d last_point_b_):
        curr_point(cur_point_),
        last_point_a(last_point_a_),
        last_point_b(last_point_b_)
        {} //cost funstion 需要传入的参数 /*当前点云中一点与上幅点云中与之最近的两点*/
        
        template <typename T>
        bool operator()(const T*q,const T *t, T *residual ) const //三维旋转与平移，二维应该就够了
        {   //cost funcstion 的公式
            Eigen::Matrix<T,3,1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
            Eigen::Matrix<T,3,1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
            Eigen::Matrix<T,3,1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

            Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};       
            
            //q_last_curr.normalize();
            Eigen::Matrix<T,3,1> t_last_curr{t[0],t[1],t[2]};
            Eigen::Matrix<T,3,1> lp;
            lp = q_last_curr *cp + t_last_curr;//quaternion overload * (pvp^-1)

            Eigen::Matrix<T,3,1> AB = lpb  - lpa;
            Eigen::Matrix<T,3,1> AC = lp -lpa ;
            Eigen::Matrix<T,3,1> abCrossac = AB.cross(AC);
            /*cout<<"abcrossac norm: "<<(abCrossac.norm())<<endl;
            cout<<"AB norm: "<<(AB.norm())<<endl;
            if(AB.norm()==abCrossac.norm())
            {
                cout<<"cp: "<<cp[0]<<cp[1]<<cp[2]<<" lpa: "<<lpa[0]<<lpa[1]<<lpa[2]<<" lpb: "<<lpb[0]<<lpb[1]<<lpb[2]<<endl;
            }*/
            residual[0] = (abCrossac.norm())/ (AB.norm()); //一个点的cost function:点线距离
            //residual[0] = AC.norm();
            return true;
        }

        static ceres::CostFunction *Create(const Eigen::Vector3d curr_point, const Eigen::Vector3d last_point_a_,
                                                                                const Eigen::Vector3d last_point_b_){
                return (new ceres::AutoDiffCostFunction<Pl_ICP,1,4,3>(new Pl_ICP(curr_point, last_point_a_, last_point_b_)));
       }
            Eigen::Vector3d curr_point, last_point_a, last_point_b;
    };

Eigen::Vector3d Quaterniond2EulerAngles(Eigen::Quaterniond q) {
  Eigen::Vector3d angles;

  // roll (x-axis rotation)
  double sinr_cosp = 2 * (q.w() * q.x() + q.y() * q.z());
  double cosr_cosp = 1 - 2 * (q.x() * q.x() + q.y() * q.y());
  angles(2) = std::atan2(sinr_cosp, cosr_cosp);

  // pitch (y-axis rotation)
  double sinp = 2 * (q.w() * q.y() - q.z() * q.x());
  if (std::abs(sinp) >= 1)
    angles(1) = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
  else
    angles(1) = std::asin(sinp);

  // yaw (z-axis rotation)
  double siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
  double cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
  angles(0) = std::atan2(siny_cosp, cosy_cosp);

  return angles;
}

//create pcl point cloud from .json file
void loadData(const string &path,
                const int &frame_index,
                Eigen::Isometry3d &T,
                pcl::PointCloud<pcl::PointXYZI> &entire_cloud,
                pcl::PointCloud<pcl::PointXYZI> &ld_cloud,
                pcl::PointCloud<pcl::PointXYZI> &pc_cloud,
                pcl::PointCloud<pcl::PointXYZI> &rb_cloud){
    Json::Reader reader;
    Json::Value root;
    
    ifstream srcFile(path, ios::binary);/*定义一个ifstream流对象，与文件demo.json进行关联*/
	if(!srcFile.is_open())
	{
		cout << "Fail to open src.json" << endl;
		return;
	}
    vector<vector<double>> coordi0;
    vector<vector<double>> coordi1;
    vector<vector<double>> coordi2;
    
    if(reader.parse(srcFile, root))
    {   
        cout<<"root.size()"<<root.size()<<endl;
        for(int i=frame_index; i<frame_index + 1/*i< root.size()*/; i++)//each frame
        {
            vector<double> position;
            for(int p=0; p<root[i]["pos"].size();p++)
            {
                position.push_back(root[i]["pos"][p].asDouble());
            }
            Eigen::Vector3d t;
            t << position[0], position[1], position[2];


            vector<double> rotation;
            for(int p=0; p<root[i]["qua"].size();p++)
            {
                rotation.push_back(root[i]["qua"][p].asDouble());
            }
            Eigen::Quaternion<double> q(rotation[3],rotation[0],rotation[1],rotation[2]);
            Eigen::Vector3d euler = Quaterniond2EulerAngles(q);
            double yaw = euler(2);
            //cout<<"yaw"<<yaw<<endl;
            Eigen::AngleAxisd rot_vect (-1*(yaw - M_PI_2), Eigen::Vector3d(0,0,1));
            T.rotate(rot_vect);
            T.pretranslate(t);

            double timestamp = root[i]["ts"].asDouble();//.asString();
            cout<<timestamp<<endl;

            for(int j=0; j<root[i]["gt"].size();j++)//each object
            {
                for(int k=0; k<root[i]["gt"][j]["x"].size();k++)
                    {
                        if(root[i]["gt"][j]["label"]==0)
                        {
                            coordi0.push_back({root[i]["gt"][j]["x"][k].asDouble(),root[i]["gt"][j]["y"][k].asDouble()});
                        }

                        if(root[i]["gt"][j]["label"]==1)
                        {
                            coordi1.push_back({root[i]["gt"][j]["x"][k].asDouble(),root[i]["gt"][j]["y"][k].asDouble()});
                        }

                        if(root[i]["gt"][j]["label"]==2)
                        {
                            coordi2.push_back({root[i]["gt"][j]["x"][k].asDouble(),root[i]["gt"][j]["y"][k].asDouble()});
                        }
                    }
            }//each object
            //break;
            ld_cloud.width = coordi0.size();
            ld_cloud.height = 1;
            ld_cloud.is_dense = false;
            ld_cloud.resize(ld_cloud.width * ld_cloud.height);
            int ldp = 0;
            for (auto& point: ld_cloud){
                point.x = coordi0[ldp][0];
                point.y = coordi0[ldp][1];
                point.z = 0;
                point.intensity = 0;
                ldp++;
            }

            pc_cloud.width = coordi1.size();
            pc_cloud.height = 1;
            pc_cloud.is_dense = false;
            pc_cloud.resize(pc_cloud.width * pc_cloud.height);
            int pcp = 0;
            for (auto& point: pc_cloud){
                point.x = coordi1[pcp][0];
                point.y = coordi1[pcp][1];
                point.z = MAX_DIS * 2;
                point.intensity = 122;
                pcp++;
            }

            rb_cloud.width = coordi2.size();
            rb_cloud.height = 1;
            rb_cloud.is_dense = false;
            rb_cloud.resize(rb_cloud.width * rb_cloud.height);
            int rbp = 0;
            for (auto& point: rb_cloud){
                point.x = coordi2[rbp][0];
                point.y = coordi2[rbp][1];
                point.z = MAX_DIS *4;
                point.intensity = 255;
                rbp++;
            }

            entire_cloud += ld_cloud;
            entire_cloud += pc_cloud; 
            entire_cloud += rb_cloud;

        }//each frame
    }


    return ;
}

void savetocsv(string path, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud){
    ofstream myfile;
    myfile.open(path);
    cout<<"width: "<<cloud->width<<" height: "<<cloud->height<<endl;
    for(int i=0; i<cloud->points.size(); ++i)
    {
        myfile << cloud->points[i].x
        <<","<<cloud->points[i].y<<"\n";
    }
    myfile.close();
    return ;
}

void runPL_ICP( string path){
                    pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("PointCloud Visualizer"));
                    visualizer->setBackgroundColor(0, 0, 0);

                    Eigen::Isometry3d cur_T = Eigen::Isometry3d::Identity();
                    pcl::PointCloud<pcl::PointXYZI> currentxyzi_cloud;
                    pcl::PointCloud<pcl::PointXYZI> cur_ld_cloud;
                    pcl::PointCloud<pcl::PointXYZI> cur_pc_cloud;
                    pcl::PointCloud<pcl::PointXYZI> cur_rb_cloud;
                    cout<<"Load scan current"<<endl;
                    loadData(path, 2, cur_T, currentxyzi_cloud, cur_ld_cloud, cur_pc_cloud, cur_rb_cloud);
                    cout<<"pose"<<"\n"<<cur_T.matrix()<<endl;

                    Eigen::Isometry3d prev_T = Eigen::Isometry3d::Identity();
                    pcl::PointCloud<pcl::PointXYZI> prev_entire_cloud;
                    pcl::PointCloud<pcl::PointXYZI> prev_ld_cloud;
                    pcl::PointCloud<pcl::PointXYZI> prev_pc_cloud;
                    pcl::PointCloud<pcl::PointXYZI> prev_rb_cloud;
                    cout<<"Load scan current"<<endl;
                    loadData(path, 0, prev_T, prev_entire_cloud, prev_ld_cloud, prev_pc_cloud, prev_rb_cloud);
                    cout<<"pose"<<"\n"<<prev_T.matrix()<<endl;
                    
                    cout<<"Start PL-ICP"<<endl;
                    pcl::PointCloud<pcl::PointXYZI>::Ptr transformedxyzi_cloud(new pcl::PointCloud<pcl::PointXYZI>());
                    Eigen::Isometry3d T_pc = Eigen::Isometry3d::Identity();
                    //Eigen::Isometry3d T_pc = cur_T.inverse()*prev_T;
                    cout<<"Transform point cloud"<<endl;
                    Eigen::Matrix4d T_gt;
                    T_gt <<   0.99983,   0.0197867,  0.00104382,  -0.0368728,
                            -0.019798,     0.99978, -0.00102556,    0.925434,
                            -0.00107379,  0.00100462,     0.99999, -0.00042957,
                            0,           0,           0,           1;
                    T_gt = T_gt.inverse().eval();
                    pcl::transformPointCloud(prev_entire_cloud,*transformedxyzi_cloud, T_pc.matrix());
                
                
                    //*transformedxyzi_cloud += pre_cloud;
                    /*for(auto point : *transformedxyzi_cloud){
                        cout<<point.x<<" "<<point.y<<" "<<point.z<<" "<<point.intensity<<endl;
                    }*/
                    /************************************************************************
                     try xyz point
                    ***********************************************************************/
                    cout<<"start transforming transformed cloud to xyz point"<<endl;
                    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
                    transformed_cloud->width = transformedxyzi_cloud->width;
                    transformed_cloud->height = 1;
                    transformed_cloud->points.resize(transformed_cloud->width*transformed_cloud->height);
                    for (int i=0; i< transformedxyzi_cloud->points.size();i++)
                    {
                        transformed_cloud->points[i].x = transformedxyzi_cloud->points[i].x;
                        transformed_cloud->points[i].y = transformedxyzi_cloud->points[i].y;
                        transformed_cloud->points[i].z = transformedxyzi_cloud->points[i].z;
                    }
                    visualizer->addPointCloud<pcl::PointXYZ>(transformed_cloud,"previous cloud");

                    cout<<"start transforming current cloud to xyz point"<<endl;
                    pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZ>());
                    current_cloud->width = currentxyzi_cloud.width;
                    current_cloud->height = 1;
                    current_cloud->points.resize(current_cloud->width*current_cloud->height);
                    for (int j=0; j< currentxyzi_cloud.points.size();j++)
                    {   
                        current_cloud->points[j].x = currentxyzi_cloud.points[j].x;
                        current_cloud->points[j].y = currentxyzi_cloud.points[j].y;
                        current_cloud->points[j].z = currentxyzi_cloud.points[j].z;
                    }
                    //visualizer->addPointCloud<pcl::PointXYZ>(current_cloud,"current cloud");
                    

                    //将local_map的点云数据输入到kdtree当中，方便后续查找
                    pcl::KdTreeFLANN<pcl::PointXYZ> pcl_kdtree_ptr_;
                    pcl_kdtree_ptr_.setInputCloud(transformed_cloud);
                    //pcl_kdtree_ptr_.setInputCloud(current_cloud);
                    
                    //将当前帧得到的二维激光点云每个点查找之前输入kdtree中的local_map最近的两个点云进行残差块的输入
                    const int point_num = current_cloud->points.size();
                    cout<<"Load settings"<<endl;
                    struct {
                    // 1. quaternion parameterization:
                    ceres::LocalParameterization *q_parameterization_ptr{nullptr};
                    // 2. loss function:
                    ceres::LossFunction *loss_function_ptr{nullptr};
                    // 3. solver:
                    ceres::Solver::Options options;
                    } config_;

                    // target variables:
                    struct {
                    double q[4] = {0.0, 0.0, 0.0, 1.0};
                    double t[3] = {0.0, 0.0, 0.0};
                    } param_;
                    ceres::Problem problem_;
                    //configs
                    {//后续判断加入的优化变量纬度是否准确
                    config_.q_parameterization_ptr = new ceres::EigenQuaternionParameterization();
                    //验证是否优化变量的纬度是否正确
                    problem_.AddParameterBlock(param_.q, 4, config_.q_parameterization_ptr);
                    problem_.AddParameterBlock(param_.t, 3);
                    config_.loss_function_ptr =  new ceres::CauchyLoss(0.1);
                    //据帧求解方法
                    config_.options.linear_solver_type = ceres::DENSE_QR;
                    // 
                    
                    config_.options.minimizer_type = ceres::TRUST_REGION;
                    config_.options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;//ceres::DOGLEG;
                    config_.options.use_nonmonotonic_steps = true;//new try
                    config_.options.initial_trust_region_radius = 1e6;//new try         
                    config_.options.dogleg_type = ceres::SUBSPACE_DOGLEG;
                    config_.options.num_threads = 8;
                    config_.options.max_num_iterations = 500;
                    config_.options.minimizer_progress_to_stdout = true;
                    config_.options.max_solver_time_in_seconds = 0.5;}

                    cout<<"formulating the pl-icp"<<endl;
                    //ros::Time start_time = ros::Time::now();
                    for(int i = 0;  i < point_num; ++i){
                        std::vector<int> corr_ind; //index
                        std::vector<float> corr_sq_dis; //
                        pcl_kdtree_ptr_.nearestKSearch(
                            current_cloud->points[i],
                            2,
                            corr_ind, corr_sq_dis
                        );
                        if(corr_sq_dis[0] > MAX_DIS){
                            continue;
                        }
                        
            
                        //TODO cerese PL-ICP
            
                        Eigen::Vector3d cp{current_cloud->points[i].x,current_cloud->points[i].y,current_cloud->points[i].z};
                        Eigen::Vector3d lpa{transformed_cloud->points[corr_ind[0]].x,transformed_cloud->points[corr_ind[0]].y,transformed_cloud->points[corr_ind[0]].z};
                        Eigen::Vector3d lpb{transformed_cloud->points[corr_ind[1]].x,transformed_cloud->points[corr_ind[1]].y,transformed_cloud->points[corr_ind[1]].z};
                        if(lpa == lpb){
                            cout<<"same point "<<corr_ind[0]<<" "<<corr_ind[1]<<endl;
                            continue;
                        }
                        ceres::CostFunction *factor_plicp = Pl_ICP::Create(
                            cp,
                            lpa,
                            lpb
                        );

                        problem_.AddResidualBlock(
                            factor_plicp,
                            nullptr,//config_.loss_function_ptr,
                            param_.q, param_.t
                        );
            
                    }
                    
                    // solve:
                    ceres::Solver::Summary summary;

                    ceres::Solve(config_.options, &problem_, &summary);
                    std::cout << " solve q , t completed " << std::endl;
                    std::cout << summary.FullReport() << "\n";

                    Eigen::Quaterniond q{param_.q[3],param_.q[0],param_.q[1],param_.q[2]};
                    Eigen::Vector3d t{param_.t[0],param_.t[1],param_.t[2]};
                    q.normalize();
                    std::cout  << " q.w() : " << q.w()<< " q.x() : " << q.x()<< " q.y() : " << q.y()<< " q.z() : " << q.z()<< std::endl;
                    cout<<"t: "<<t[0]<<" "<<t[1]<<" "<<t[2]<<endl;

                    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>());
                    Eigen::Isometry3d out_T = Eigen::Isometry3d::Identity();
                    out_T.rotate(q);
                    out_T.pretranslate(t);
                    //Eigen::Matrix4d final_T = out_T.matrix()*T_gt;
                    pcl::transformPointCloud(*current_cloud,*output_cloud,out_T.matrix());
                    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(output_cloud, 0, 255, 0);
                    visualizer->addPointCloud<pcl::PointXYZ>(output_cloud,single_color,"output cloud");
                    //ros::Time end_time = ros::Time::now();
                    //ros::Duration use_time = end_time - start_time;
                    //std::cout << " use time : " << use_time.toSec() << std::endl;

                    visualizer->setPointCloudRenderingProperties(
                    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "PointCloud");
                    visualizer->addCoordinateSystem(5.0);

                    while (!visualizer->wasStopped()) {
                    visualizer->spinOnce(100);}
                    return;
}

void runICP(string path){
    cout<<"start ICP"<<endl;
    pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("PointCloud Visualizer"));
    visualizer->setBackgroundColor(0, 0, 0);
    pcl::PointCloud<pcl::PointXYZI>::Ptr global_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cur_ld_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cur_pc_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cur_rb_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Matrix4d T = Eigen::Isometry3d::Identity().matrix();
    deque<pcl::PointCloud<pcl::PointXYZI>> window;
    Eigen::Isometry3d init_T = Eigen::Isometry3d::Identity();
    for(int i = 0; i< 20; i++)
    {
        Eigen::Isometry3d cur_T = Eigen::Isometry3d::Identity();
        pcl::PointCloud<pcl::PointXYZI> cur_entire_cloud;
        pcl::PointCloud<pcl::PointXYZI> cur_ld_cloud;
        pcl::PointCloud<pcl::PointXYZI> cur_pc_cloud;
        pcl::PointCloud<pcl::PointXYZI> cur_rb_cloud;
        cout<<"Load scan "<<i<<endl;
        loadData(path, i, cur_T, cur_entire_cloud, cur_ld_cloud, cur_pc_cloud, cur_rb_cloud);
        cout<<"pose"<<"\n"<<cur_T.matrix()<<endl;
        pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
        icp.setMaximumIterations(100);
        icp.setMaxCorrespondenceDistance(MAX_DIS);
        if(i==0)
        {
            init_T = cur_T;
            *global_cloud = cur_entire_cloud;
            *cur_ld_cloud_ptr = cur_ld_cloud;
            *cur_pc_cloud_ptr = cur_pc_cloud;
            *cur_rb_cloud_ptr = cur_rb_cloud;
            window.push_back(cur_entire_cloud);
            //visualizer->addPointCloud<pcl::PointXYZI>(global_cloud,"current cloud");
        }
        else
        {
            cur_T = init_T.inverse() * cur_T; 
            pcl::PointCloud<pcl::PointXYZI>::Ptr CarPosetrans_cloud(new pcl::PointCloud<pcl::PointXYZI>);      
            cout<<"car pose: "<<"\n"<<cur_T.matrix()<<endl;      
            pcl::transformPointCloud(cur_entire_cloud,*CarPosetrans_cloud,cur_T.matrix());
            pcl::PointCloud<pcl::PointXYZI>::Ptr local_map(new pcl::PointCloud<pcl::PointXYZI>);
            for(int j =0;j<window.size();j++)
            {
                *local_map += window[j];
            }
            pcl::PointCloud<pcl::PointXYZI>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            //pcl::transformPointCloud(pre_entire_cloud,*transformedxyzi_cloud,(Eigen::Isometry3d::Identity()).matrix());
            *source_cloud += cur_entire_cloud;
            icp.setInputSource(CarPosetrans_cloud);//source_cloud);
            icp.setInputTarget(local_map);
            icp.align(*CarPosetrans_cloud);//source_cloud);
            //bool icp_succeeded_or_not = icp.hasConverged();
            //double match_score = icp.getFitnessScore();
            Eigen::Matrix4d transform_pred = icp.getFinalTransformation().cast<double>();
            //T = T * transform_pred;
            cout<<"transform predict: "<<"\n"<<transform_pred<<endl;
            //string name = "cloud " + to_string(i);
            //pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            //pcl::transformPointCloud(cur_entire_cloud,*transformed_cloud,transform_pred);
            //visualizer->addPointCloud<pcl::PointXYZI>(transformed_cloud,name);
            *global_cloud += *CarPosetrans_cloud;//*source_cloud;//*transformed_cloud;
            if(window.size()>=20)
            {
                window.pop_front();
            }
            window.push_back(*CarPosetrans_cloud);
                
        }
        
    }
    visualizer->addPointCloud<pcl::PointXYZI>(global_cloud);
    visualizer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "PointCloud");
    visualizer->addCoordinateSystem(5.0);

    while (!visualizer->wasStopped()) {
    visualizer->spinOnce(100);}

}

void windowCurFit(string path){
    pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("PointCloud Visualizer"));
    visualizer->setBackgroundColor(0, 0, 0);
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr global_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr global_ld_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr global_pc_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr global_rb_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Isometry3d init_T = Eigen::Isometry3d::Identity();
    for(int i = 0; i< 20; i++)
    {
        Eigen::Isometry3d cur_T = Eigen::Isometry3d::Identity();
        pcl::PointCloud<pcl::PointXYZI> cur_entire_cloud;
        pcl::PointCloud<pcl::PointXYZI> cur_ld_cloud;
        pcl::PointCloud<pcl::PointXYZI> cur_pc_cloud;
        pcl::PointCloud<pcl::PointXYZI> cur_rb_cloud;
        cout<<"Load scan "<<i<<endl;
        loadData(path, i, cur_T, cur_entire_cloud, cur_ld_cloud, cur_pc_cloud, cur_rb_cloud);
        cout<<"pose"<<"\n"<<cur_T.matrix()<<endl;
        if(i==0)
        {
            init_T = cur_T;
            *global_cloud = cur_entire_cloud;
            *global_ld_cloud_ptr = cur_ld_cloud;
            *global_pc_cloud_ptr = cur_pc_cloud;
            *global_rb_cloud_ptr = cur_rb_cloud;
        }
        else
        {
            cur_T = init_T.inverse() * cur_T; 
            pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_ld_cloud(new pcl::PointCloud<pcl::PointXYZI>);      
            cout<<"car pose: "<<"\n"<<cur_T.matrix()<<endl;      
            pcl::transformPointCloud(cur_ld_cloud,*transformed_ld_cloud,cur_T.matrix());
            *global_ld_cloud_ptr += *transformed_ld_cloud;
        }        
    }
    vector<vector<int>> clusters_index;
    dbscan<pcl::PointXYZI>(global_ld_cloud_ptr,clusters_index,2.0,5);
    string csvpath = "/root/bev_fusion/pl_icp/lanecloud.csv";
    savetocsv(csvpath, global_ld_cloud_ptr);
    int color = 255 / (clusters_index.size() + 5);
    for(int i=0; i< clusters_index.size(); ++i)
    {
        sort(clusters_index[i].begin(),clusters_index[i].end());
        pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        for(auto index : clusters_index[i])
        {
            *cluster_cloud += *global_ld_cloud_ptr;
        }      
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> single_color(cluster_cloud, (i+1)*color, (i+1)*color, (i+1)*color);
        pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZI> rgb(cluster_cloud);
        string name = "cloud " + to_string(i);
        visualizer->addPointCloud<pcl::PointXYZI>(cluster_cloud,rgb, name);

    }
    visualizer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "PointCloud");
    visualizer->addCoordinateSystem(5.0);

    while (!visualizer->wasStopped()) {
    visualizer->spinOnce(100);}

    
}

int main(){
    string path = "/root/bev_fusion/pl_icp/output_demo.json";
    //runICP(path);
    windowCurFit(path);

    return 0;
}
