#include <iostream>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>	//	pcl::transformPointCloud 用到这个头文件
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <json/json.h>
#include <json/value.h>
#include <fstream>

#include <vector>
#include <string>

using namespace Eigen;
using namespace std;

#define MAX_DIS 0.25

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
            Eigen::Matrix<T, 3, 1> lp;
            lp = q_last_curr *cp + t_last_curr;

            Eigen::Matrix<T,3,1> AB = lpb  - lpa;
            Eigen::Matrix<T,3,1> AC = lp -lpa ;
            Eigen::Matrix<T,3,1> abCrossac = AB.cross(AC);
             residual[0] = (abCrossac.norm())/ (AB.norm()); //一个点的cost function:点线距离
            
            return true;
        }

        static ceres::CostFunction *Create(const Eigen::Vector3d curr_point, const Eigen::Vector3d last_point_a_,
                                                                                const Eigen::Vector3d last_point_b_){
                return (new ceres::AutoDiffCostFunction<Pl_ICP,1,4,3>(new Pl_ICP(curr_point, last_point_a_, last_point_b_)));
       }
            Eigen::Vector3d curr_point, last_point_a, last_point_b;
    };

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
            Eigen::Quaternion<double> q(Eigen::Vector4d(rotation[0],rotation[1],rotation[2],rotation[3]));
            T.rotate(q);
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
                point.z = 0;
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
                point.z = 0;
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


void runPl_ICP( Eigen::Quaterniond &pred_q,
                 Eigen::Vector3d &pred_t,
                 const Eigen::Isometry3d &T_pc, 
                pcl::PointCloud<pcl::PointXYZI> &pre_cloud,
                pcl::PointCloud<pcl::PointXYZI> &currentxyzi_cloud){
                cout<<"Start PL-ICP"<<endl;
                pcl::PointCloud<pcl::PointXYZI>::Ptr transformedxyzi_cloud(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::KdTreeFLANN<pcl::PointXYZ> pcl_kdtree_ptr_;
                //将local_map的点云数据输入到kdtree当中，方便后续查找
                /*for(int i = 0; i < pre_cloud.size(); ++i){
                        pcl::transformPointCloud(
                            pre_cloud.at(i).frame_, 
                            *transformed_cloud_ptr, 
                            pre_cloud.at(i).pose
                        );
                        *map_ptr_ += *transformed_cloud_ptr;
                    }*/
                cout<<"Transform point cloud"<<endl;
                pcl::transformPointCloud(pre_cloud,*transformedxyzi_cloud,T_pc.matrix());
                /*for(auto point : currentxyzi_cloud){
                    cout<<point.x<<" "<<point.y<<" "<<point.z<<" "<<point.intensity<<endl;
                }*/
                /************************************************************************
                 try xyz point
                ***********************************************************************/
                cout<<"start transforming xyz point"<<endl;
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
                cout<<"transformed xyz completed"<<endl;
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
                
                pcl_kdtree_ptr_.setInputCloud(transformed_cloud);
                //pcl_kdtree_ptr_.setInputCloud(current_cloud);
                cout<<"setInputCloud no segmentation error"<<endl;
                
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
                //后续判断加入的优化变量纬度是否准确
                config_.q_parameterization_ptr = new ceres::EigenQuaternionParameterization();
                //损失函数
                
            
                //验证是否优化变量的纬度是否正确
                problem_.AddParameterBlock(param_.q, 4, config_.q_parameterization_ptr);
                problem_.AddParameterBlock(param_.t, 3);
                config_.loss_function_ptr =  new ceres::CauchyLoss(0.1);
                //据帧求解方法
                config_.options.linear_solver_type = ceres::DENSE_QR;
                // 
                
                config_.options.minimizer_type = ceres::TRUST_REGION;
                config_.options.trust_region_strategy_type = ceres::DOGLEG;
                config_.options.dogleg_type = ceres::SUBSPACE_DOGLEG;
                config_.options.num_threads = 8;
                config_.options.max_num_iterations = 100;
                config_.options.minimizer_progress_to_stdout = false;
                config_.options.max_solver_time_in_seconds = 0.10;

                
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
                    ceres::CostFunction *factor_plicp = Pl_ICP::Create(
                        cp,
                        lpa,
                        lpb
                    );

                    problem_.AddResidualBlock(
                        factor_plicp,
                        config_.loss_function_ptr,
                        param_.q, param_.t
                    );
        
                }
                
                // solve:
                ceres::Solver::Summary summary;

                ceres::Solve(config_.options, &problem_, &summary);
                std::cout << " solve q , t completed " << std::endl;
                
                Eigen::Quaterniond q{param_.q[3],param_.q[0],param_.q[1],param_.q[2]};
                Eigen::Vector3d t{param_.t[0],param_.t[1],param_.t[2]};
                q.normalize();
                std::cout  << " q.w() : " << q.w()<< " q.x() : " << q.x()<< " q.y() : " << q.y()<< " q.z() : " << q.z()<< std::endl;
                pred_q = q;
                pred_t = t;
                cout<<t[0]<<t[1]<<t[2]<<endl;

                //ros::Time end_time = ros::Time::now();
                //ros::Duration use_time = end_time - start_time;
                //std::cout << " use time : " << use_time.toSec() << std::endl;
                return;
}

int main(){
    string path = "/root/bev_fusion/pl_icp/output_demo.json";
    int frmdx;
    /********************************************************
     try kdtree
    ********************************************************/
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);    //创建一个指向点云的指针
     /**************************生成点云方法1**************************/
	//初始化点云相关变量
	cloud->width = 1000;    
	cloud->height = 1;  //说明是无序点云
	cloud->points.resize(cloud->width* cloud->height);  //点云的点数

	//生成点云
	for (std::size_t t = 0; t < cloud->size(); t++)
	{
		cloud->points[t].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
		cloud->points[t].y = 1024.0f * rand() / (RAND_MAX + 1.0f);
		cloud->points[t].z = 1024.0f * rand() / (RAND_MAX + 1.0f);
	}
	/****************************生成点云方法2*************************/
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointXYZ point;
	for (std::size_t t = 0; t < 1000; t++)
	{
		point.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
		point.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
		point.z = 1024.0f * rand() / (RAND_MAX + 1.0f);
		cloud_1->push_back(point);       //采用push_back添加点进而生成点云
	}
	//创建KD-Tree实例
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);
    cout<<"set input cloud can work"<<endl;
	//随机初始化一个查找点
	pcl::PointXYZ searchPoint; 
    /********************************************************
     try kdtree
    ********************************************************/

    
    frmdx = 15;
    Eigen::Isometry3d cur_T = Eigen::Isometry3d::Identity();
    pcl::PointCloud<pcl::PointXYZI> cur_entire_cloud;
    pcl::PointCloud<pcl::PointXYZI> cur_ld_cloud;
    pcl::PointCloud<pcl::PointXYZI> cur_pc_cloud;
    pcl::PointCloud<pcl::PointXYZI> cur_rb_cloud;
    cout<<"Load current scan"<<endl;
    loadData(path, frmdx, cur_T, cur_entire_cloud, cur_ld_cloud, cur_pc_cloud, cur_rb_cloud);
    cout<<"current pose"<<"\n"<<cur_T.matrix()<<endl;

    //previous scan
    frmdx = 0;
    Eigen::Isometry3d pre_T = Eigen::Isometry3d::Identity();
    pcl::PointCloud<pcl::PointXYZI> pre_entire_cloud;
    pcl::PointCloud<pcl::PointXYZI> pre_ld_cloud;
    pcl::PointCloud<pcl::PointXYZI> pre_pc_cloud;
    pcl::PointCloud<pcl::PointXYZI> pre_rb_cloud;
    cout<<"Load previous scan"<<endl;
    loadData(path, frmdx, pre_T, pre_entire_cloud, pre_ld_cloud, pre_pc_cloud, pre_rb_cloud);
    cout<<"previous pose"<<"\n"<<pre_T.matrix()<<endl;

    Eigen::Quaterniond q;
    Eigen::Vector3d t;
    Eigen::Isometry3d T_pc; //= Eigen::Isometry3d::Identity();
    T_pc = cur_T*pre_T.inverse();
    cout<<"T_pc"<<"\n"<<T_pc.matrix()<<endl;
    runPl_ICP(q, t, T_pc, pre_entire_cloud, cur_entire_cloud);
    return 0;
}