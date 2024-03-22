// Implementation of the KMeans Algorithm
// reference: https://github.com/marcoscastro/kmeans
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include "oneapi/tbb/parallel_reduce.h"
#include <tbb/global_control.h>
#include <tbb/task.h>
#include <tbb/parallel_reduce.h>
#include <tbb/spin_mutex.h>
#include <random>
#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <list>
#include <atomic>
#include <tuple>
#include <map>
using namespace std;


class Point
{
private:
	int id_point, id_cluster;
	vector<double> values;
	int total_values;
	string name;

public:
	Point(int id_point, vector<double>& values, string name = "")
	{
		this->id_point = id_point;
		total_values = values.size();

		for(int i = 0; i < total_values; i++)
			this->values.push_back(values[i]);

		this->name = name;
		id_cluster = -1;
	}

	int getID()
	{
		return id_point;
	}

	void setCluster(int id_cluster)
	{
		this->id_cluster = id_cluster;
	}

	int getCluster()
	{
		return id_cluster;
	}

	double getValue(int index)
	{
		return values[index];
	}

	int getTotalValues()
	{
		return total_values;
	}

	void addValue(double value)
	{
		values.push_back(value);
	}

	string getName()
	{
		return name;
	}
};


class Cluster
{
private:
	int id_cluster;
	vector<double> central_values;
	vector<Point> points;
public:
	Cluster(int id_cluster, Point point)
	{
		this->id_cluster = id_cluster;

		int total_values = point.getTotalValues();

		// For all values in the current point, add it to the central vector
		for(int i = 0; i < total_values; i++){
			central_values.push_back(point.getValue(i));
		}
		// Then add the point to the vector of all points
		points.push_back(point);

	}

	void addPoint(Point point)
	{
		points.push_back(point);
	}

	bool removePoint(int id_point)
	{
		int total_points = points.size();

		for(int i = 0; i < total_points; i++)
		{
			if(points[i].getID() == id_point)
			{
				points.erase(points.begin() + i);
				return true;
			}
		}
		return false;
	}

	double getCentralValue(int index)
	{
		return central_values[index];
	}

	void setCentralValue(int index, double value)
	{
		central_values[index] = value;
	}

	Point getPoint(int index)
	{
		return points[index];
	}

	int getTotalPoints()
	{
		return points.size();
	}

	int getID()
	{
		return id_cluster;
	}

};


class KMeans
{
private:
	int K; // number of clusters
	int total_values, total_points, max_iterations;
	vector<Cluster> clusters;
 
	int getIDNearestCenterPar(Point point)
	{
		
		// Holds the squared differences and the smallest dist found
		double sum = 0.0; 
		double min_dist = 0.0;

		int id_cluster_center = 0;

		for(int i = 0; i < total_values; i++)
		{
			sum += pow(clusters[0].getCentralValue(i) - point.getValue(i), 2.0);
		}

		// Square root of the sum of distances
		min_dist = sqrt(sum);

		// Create a map to store distances and their corresponding cluster indices
		std::map<double, int> distance_map;

		// Iterate over all other cluster centers and store their distances
		for(int i = 1; i < K; i++)
		{
			sum = 0.0;

			for(int j = 0; j < total_values; j++)
			{
				sum += pow(clusters[i].getCentralValue(j) - point.getValue(j), 2.0);
			}

			double dist = sqrt(sum);
			distance_map[dist] = i; // Store the distance and its corresponding cluster index
		}

		// Find the iterator to the minimum distance
		auto min_it = distance_map.begin();

		// Retrieve the index of the nearest cluster
		int min_index = min_it->second;

		// Return the closest center
    	return min_index;
	}
	
	// Change Sunlab Machine: ssh (name)
	// Return ID of nearest center (uses euclidean distance)
	int getIDNearestCenter(Point point)
	{	
		// Holds the squared differences and the smallest dist found
		double sum = 0.0, 
		min_dist;

		int id_cluster_center = 0;
		

		// Accumulate all the squared euclidean distances between the values and the first cluster
		//One iteration per point
		for(int i = 0; i < total_values; i++)
		{
			sum += pow(clusters[0].getCentralValue(i) - point.getValue(i), 2.0);
		}

		// Square root of the sum of distances
		min_dist = sqrt(sum);

		// Iterate over all other cluster centers and do the same thing
		for(int i = 1; i < K; i++)
		{
			double dist;
			sum = 0.0;


			for(int j = 0; j < total_values; j++)
			{
				sum += pow(clusters[i].getCentralValue(j) - point.getValue(j), 2.0);
			}

			dist = sqrt(sum);

			// Keep updating the shortest distance and store whatever center it's associated to
			if(dist < min_dist)
			{
				min_dist = dist;
				id_cluster_center = i;
			}
		}

		// Return the closest center
		return id_cluster_center;
	}

public:
	KMeans(int K, int total_points, int total_values, int max_iterations)
	{
		// Number of clusters
		this->K = K;

		// Number of points to cluster
		this->total_points = total_points;

		// Values of each data point
		this->total_values = total_values;

		// Total number of iterations allowed
		this->max_iterations = max_iterations;

	}

	// Generate a random integer
	int rand_int(int min, int max){

		std::random_device rd;
		std::mt19937 gen(rd());

		std::uniform_int_distribution<> rand_dist(min, max);
		int rand_int = rand_dist(gen);

		return rand_int;
	}
	
	// Compute the K-Means algorithm
	void run(vector<Point> & points)
	{
        auto begin = chrono::high_resolution_clock::now();
        
		// Can't have more clusters than total points
		if(K > total_points){
			return;
		}
		
		vector<int> prohibited_indexes;
		vector<int> indices;
		
		// ------------------------------------- Good cluster centers ------------------------------------- //

		// Raisins 
		//std::vector<int> good = {408, 668, 284};
		//std::vector<int> good = {487, 448, 70, 663};
		//std::vector<int> good = {645, 843, 614, 586, 613};
		//std::vector<int> good = {784, 541, 810, 855, 793, 80};
		//std::vector<int> good = {329, 542, 781, 234, 869, 365, 506};
		//std::vector<int> good = {750, 569, 714, 260, 680, 599, 892, 245, 259, 860};
		//std::vector<int> good = {750, 569, 714, 260, 680, 599, 892, 245, 259, 860, 853, 421, 228, 34, 738};


		// Red Wine
		//std::vector<int> good = {391, 173, 886};
		//std::vector<int> good = {1074, 981, 1289, 361};
		//std::vector<int> good = {1063, 407, 514, 154, 361};
		//std::vector<int> good = {1399, 192, 861, 1063, 1452, 881};
		//std::vector<int> good = {154, 918, 596, 81, 1471, 1138, 1532};
		//std::vector<int> good = {1070, 172, 159, 828, 506, 923, 571, 885, 1154, 1444};
		//std::vector<int> good = {154, 698, 488, 1565, 373, 514, 664, 940, 1367, 459, 859, 1267, 407, 559, 1335};

		
		// Urban
		//std::vector<int> good = {142945, 150577, 279707};
		//std::vector<int> good = {310389, 289875, 192130, 268307};
		//std::vector<int> good = {124293, 151558, 282776, 197587, 320378};
		//std::vector<int> good = {168617, 134106, 49603, 45522, 210675, 30824};
		//std::vector<int> good = {198905, 206541, 249272, 85957, 295038, 233812, 279669};
		//std::vector<int> good = {187896, 324141, 53841, 315771, 356793, 177618, 344746, 26313, 318320, 204329};
		//std::vector<int> good = {198505, 251581, 282541, 121544, 182451, 40306, 131317, 218715, 70986, 136454, 302799, 19219, 58509, 50956, 248408};

		// Bank
		std::vector<int> good = {1008, 5196, 3037};
		//std::vector<int> good = {5383, 5069, 2483, 2487};
		//std::vector<int> good = {5037, 3246, 2778, 3510, 3001};
		//std::vector<int> good = {564, 5902, 5023, 5864, 5758, 1177};
		//std::vector<int> good = {5716, 495, 6114, 5678, 6682, 6608, 72};
		//std::vector<int> good = {794, 5198, 2353, 4907, 1344, 680, 5989, 4364, 5359, 2445};
		//std::vector<int> good = {5884, 4617, 1674, 2319, 4727, 4563, 6489, 2530, 2311, 3149, 1119, 4387, 4906, 5296, 458};

		// ------------------------------------------------------------------------------------------------ //
		
		int good_point;
		for(int i=0; i < K; ++i){
			good_point = good[i];
			points[good_point].setCluster(i);
			Cluster cluster(i, points[good_point]);
			clusters.push_back(cluster);
			indices.push_back(good_point);
		}

		// Choose random cluster centers
		// for(int i = 0; i < K; i++)
		// {
		// 	while(true)
		// 	{
		// 		int index_point = rand_int(0, total_points) % total_points;

		// 		if(find(prohibited_indexes.begin(), prohibited_indexes.end(), index_point) == prohibited_indexes.end())
		// 		{
		// 			prohibited_indexes.push_back(index_point);
		// 			points[index_point].setCluster(i);

		// 			Cluster cluster(i, points[index_point]);
		// 			clusters.push_back(cluster);
		// 			break;
		// 		}
		// 	}
			
		// }

        auto end_phase1 = chrono::high_resolution_clock::now();

		auto begin_first = chrono::high_resolution_clock::now();
		auto begin_second = chrono::high_resolution_clock::now();
		auto end_first = chrono::high_resolution_clock::now();
		auto end_second = chrono::high_resolution_clock::now();

		int iter = 1;
		std::vector<std::mutex> sums_mutexes(K);
		std::mutex tot_points_mutex;

		while(true)
		{
			// Tracks if any point has moved to a different cluster
			// Once this remains true, the algorithm has converged and we're done
			bool done = true;
			

			// -------------------- Parallel Implementation -------------------- // 

			begin_second = chrono::high_resolution_clock::now();

			std::vector<int> tot_points(K, 0);
			
			// This contains a vector for each cluster
			// Each inner vector will hold the sums for each value of a cluster
			std::vector<std::vector<double>> sums(K, std::vector<double>(total_values));

			tbb::parallel_for(tbb::blocked_range<int>(0, total_points), [&](const tbb::blocked_range<int>& r) {
					
					for (int i = r.begin(); i != r.end(); ++i) {
						
						// The ID of the cluster the point is assigned to
						int id_old_cluster = points[i].getCluster();

						// Find the closest centroid
						int id_nearest_center = getIDNearestCenter(points[i]);

						// If the current center isn't the closest, change it
						if(id_old_cluster != id_nearest_center)
						{
							// Assign point to the new closest
							points[i].setCluster(id_nearest_center);

							// Keep iterating since we haven't converged
							done = false;
						}

						sums_mutexes[id_nearest_center].lock();
						for (int k = 0; k < total_values; k++) {
							sums[id_nearest_center][k] += points[i].getValue(k);
						}
						sums_mutexes[id_nearest_center].unlock();

						// Protect access to tot_points vector
						tot_points_mutex.lock();
						tot_points[id_nearest_center] += 1;
						tot_points_mutex.unlock();
						
					}
        	});
			
			// for(int i = 0; i < total_points; i++)
			// {	
			// 	// The ID of the cluster the point is assigned to
			// 	int id_old_cluster = points[i].getCluster();

			// 	// Find the closest centroid
			// 	int id_nearest_center = getIDNearestCenter(points[i]);

			// 	// If the current center isn't the closest, change it
			// 	if(id_old_cluster != id_nearest_center)
			// 	{
			// 		// Assign point to the new closest
			// 		points[i].setCluster(id_nearest_center);

			// 		// Keep iterating since we haven't converged
			// 		done = false;
			// 	}

			// 	// Add the current point's values to the corresponding cluster vector
			// 	for(int k=0; k<total_values; k++){
			// 		sums[id_nearest_center][k] += points[i].getValue(k);
			// 	}

			// 	// Keep track of how many points are in each cluster
			// 	tot_points[id_nearest_center] += 1;
			// }

			// Loop over all clusters
			for (int i=0; i<K; ++i) {
				// Loop over all vals
				for(int j = 0; j < total_values; j++){	
					clusters[i].setCentralValue(j, sums[i][j] / tot_points[i]);
				}
			}


			end_second = chrono::high_resolution_clock::now();


			// Exit if we hit a terminating condition
			if(done == true || iter >= max_iterations)
			{
				cout << "Break in iteration " << iter << "\n\n";
				break;
			}
			iter++;
		}
		
        auto end = chrono::high_resolution_clock::now();

		// Shows elements of clusters
		for(int i = 0; i < K; i++)
		{
	
			//  Print value of all centers
			cout << "Cluster values: ";
			for(int j = 0; j < total_values; j++){
				cout << clusters[i].getCentralValue(j) << " ";
			}

			// Print execution time
			cout << "\n\n";
            cout << "TOTAL EXECUTION TIME = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"\n";
            
            cout << "TIME PHASE 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"\n";
            
            cout << "TIME PHASE 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_second-begin_second).count()<<"\n";

		}
	}
};

int main(int argc, char *argv[])
{
	srand (time(NULL));

	// Input parameters
	// 		points: total number of data points to be organized into clusters
	// 		values: total number of dimensions for every point
	// 		K: number of clusters to be made
	// 		iterations: max number of iterations allowed for the K-means algorithm
	// 		name: indicates whether there's a name for a data point
	int total_points, total_values, K, max_iterations, has_name;

	cin >> total_points >> total_values >> K >> max_iterations >> has_name;

	// Vector of Point class
	vector<Point> points;
	
	string point_name;
	//int num_threads = 2;

	// Loop over all points
	for(int i = 0; i < total_points; i++)
	{
		// Create a vector of floats to hold the values for each point
		vector<double> values;

		// For all values for the current point, add them to the vector
		for(int j = 0; j < total_values; j++)
		{
			double value;
			cin >> value;
			values.push_back(value);
		}

		// If given a name, create a new point with the name
		if(has_name)
		{
			cin >> point_name;
			Point p(i, values, point_name);
			points.push_back(p);
		}
		// If not, create the point with just ID and values vector
		else
		{
			Point p(i, values);
			points.push_back(p);
		}
	}

	// After all points are added to the data structures,
	// create a new KMeans instance and run the algorithm on all points
	KMeans kmeans(K, total_points, total_values, max_iterations);
	kmeans.run(points);
	
	return 0;
}