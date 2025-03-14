#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

int main(int argc, char** argv)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

	// Fill in the cloud data
	pcl::PCDReader reader;
	reader.read("../../model/fingertip.pcd", *cloud);
	std::cerr << "Cloud before filtering: " << cloud->points.size() << std::endl;
	// Create the filtering object
	pcl::PassThrough<pcl::PointXYZ> pass; // 声明直通滤波
	pass.setInputCloud(cloud); // 传入点云数据
	pass.setFilterFieldName("y"); // 设置操作的坐标轴
	pass.setFilterLimits(-100, 100); // 设置坐标范围  #0~5.0
	// pass.setFilterLimitsNegative(true); // 保留数据函数
	pass.filter(*cloud_filtered);  // 进行滤波输出

    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-100, 100); //27.0~30.0
    pass.filter(*cloud_filtered);

    pass.setInputCloud(cloud_filtered);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(12, 100); //-2~2
    pass.filter(*cloud_filtered);


    // pass.setInputCloud(cloud_filtered);
    // pass.setFilterFieldName("x");
    // pass.setFilterLimits(0, 50.0);
    // pass.filter(*cloud_filtered);

	std::cerr << "Cloud after filtering: " << cloud_filtered->points.size() << std::endl;

	// save filterd data
	pcl::PCDWriter writer;
	writer.write("../../model/fingertip_part.pcd", *cloud_filtered, false);

	return 0;
}
