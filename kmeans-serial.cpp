// Implementation of the KMeans Algorithm
// reference: https://github.com/marcoscastro/kmeans

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <tbb/tbb.h>
#include <mutex>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_reduce.h>
using namespace std;

// struct sum_and_count {
// 	sum_and_count() : sum(), count(0) {} point sum;
// 	size_t count;

// 	void clear() {
// 		sum = point(); 
// 		count = 0;
// 	}

// 	void tally( const point& p ) { 
// 		sum += p;
// 		++count;
// 	}

// 	point mean() const {
// 		return sum/count;
// 	}

// 	// Combines the results from multiple instances of this struct
// 	void operator+=( const sum_and_count& other ) {
// 		sum += other.sum; 
// 		count += other.count;
// 	} 
// };

// class view 
// {
// 	view(const view& v);               // Deny copy construction to remain thread safe
// 	void operator=( const view& v );   // Deny assignment

// public:

// 	// Array of sum & count objects
// 	// Each object represents the sum & count of points of a certain cluster
// 	// This holds the local sums of points for each cluster
// 	sum_and_count* array;

// 	// Tracks total number of changes made across all threads
// 	size_t change;

// 	// Initializes each sum and count object to hold the sum & count of points for each cluster
// 	view( size_t k ) : array(new sum_and_count[k]), change(0) {}  ̃view() {delete[] array;}
// };

// // Implements a collection of thread local views
// // I'm assuming you use this to make as many thread views as clusters?
// typedef tbb::enumerable_thread_specific<view> tls_type;


// // Parameters:
// //		tls: reference to a TLS object containing views of local sums for each thread
// //		global: global view containing the combined sums of points for clusters across all threads
// void reduce_local_counts_to_global_count( tls_type& tls, view& global ) { 

// 	// Initialize changes as 0
// 	global.change = 0;

// 	// Iterate over the TLS
// 	for( auto i=tls.begin(); i!=tls.end(); ++i ) {
// 		// Dereference the iterator & get a reference to current TLS view
// 		view& v = *i; 

// 		// Add all change variables from all threads to the global change tracker
// 		global.change += v.change; 

// 		// Clear the local changes for the next K-means iteration
// 		v.change = 0;
// 	}
// }


// // Parameters:
// //		k: number of clusters
// //		tls: reference to a TLS object containing views of local sums for each thread
// //		global: global view containing the combined sums of points for clusters across all threads
// void reduce_local_sums_to_global_sum( size_t k, tls_type& tls, view& global ) { 

// 	// Iterate over the thread-local storage
// 	for( auto i=tls.begin(); i!=tls.end(); ++i ) {
// 		// Dereference the iterator & get a reference to current TLS view
// 		view& v = *i;

// 		// Loop over each cluster
// 		for( size_t j=0; j<k; ++j ) {
// 			// Add the sums of points in the TL view of a cluster to
// 			// its corresponding sum in the global view
// 			global.array[j] += v.array[j]; 

// 			// Clear the TL view for the next iteration
// 			v.array[j].clear();
// 		}
// 	}
// }


// // Parameters:
// //		centroid: vector of all centroids
// //		k: number of centroids
// //		value: looking for the center closest to this point
// int reduce_min_ind( const point centroid[], size_t k, point value ) { 

// 	// Index of the closest centroid
// 	int min = -1;

// 	// Minimum distance squared
// 	float mind = std::numeric_limits<float>::max();

// 	// Loop over all centers
// 	for( int j=0; j<k; ++j ) {

// 		// Get squared euclidean distance between the point & current center
// 		float d = distance2(centroid[j], value); 

// 		// If we find a new minimum distance, set it and save the index of the center
// 		if( d<mind ) {
// 			mind = d; 
// 			min = j;
// 		}
// 	}

// 	// Return the closest center's index
// 	return min;
// }

// // Parameters:
// //		n: number of data points
// //		points: vector of all points on the graph
// //		k: number of clusters
// //		id: ID of a cluster
// //		centroid: stores all cluster centers
// void compute_k_means( size_t n, const Point points[], size_t k, cluster_id id[], point centroid[] ) {

// 	// Initialize a new thread-local storage object
// 	// This allows a thread to have its own independent storage
// 	tls_type tls([&]{return k;}); 

// 	// Create a view of data shared across all threads
// 	// Used to combine data from each thread e.g. the number of points in a cluster
// 	view global(k);

// 	// Create initial clusters and compute their sums
// 	tbb::parallel_for(
// 		// Runs "n" number of times
// 		tbb::blocked_range<size_t>(0,n),
// 		// Lambda:
// 		//	=: captures everything by value (modifications within won't affect it)
// 		//  &tls: captures the thread-local-storage by reference (can modify the object)
// 		//  &global: captures the view by reference (can modify it)		
// 		[=, &tls, &global]( tbb::blocked_range<size_t> r ) {

// 			// Get the TLS of the current thread
// 			// view is used as a container to hold all the thread's data
// 			view& v = tls.local();

// 			// Iterate over the subrange of indices assigned to the current thread
// 			for( size_t i=r.begin(); i!=r.end(); ++i ) {

// 				// Based on index, assign a cluster ID to each point
// 				id[i] = i % k; 

// 				// Peeled “Sum step”
// 				// 		Update the sums of points associated with each cluster in the thread-local view
// 				//		Calls the 'tally' method of the sum & count struct
// 				//		Tally adds the features of the ith point to the sum of clusters indicated by point id[i]
// 				v.array[id[i]].tally(points[i]);
// 			} 
// 		}
// 	);


// 	// Loop until ids do not change
// 	size_t change; 
//	do{

// 		// Add local sums to global sum and clear TLS
//    		reduce_local_sums_to_global_sum( k, tls, global );

// 		// Repair any empty clusters
//    		repair_empty_clusters(n, points, id, k, centroid, global.array);

// 		// “Divide step”: compute centroids from global sums
// 		// Loop over all cluster centers
// 		for( size_t j=0; j<k; ++j ) { 
// 			// Take the mean of the points assigned to that cluster
// 			// This uses the 'mean' method of the sum & count struct
// 			// that is stored in global.array[j]
// 			centroid[j] = global.array[j].mean(); 

// 			// Once the centroid is calcuated, clear all sum & count objects
// 			// to reset everything for the next iteration
// 			global.array[j].clear();
// 		}

//         // Compute new clusters and their local sums
// 		tbb::parallel_for(
// 			tbb::blocked_range<size_t>(0,n),
// 			[=, &tls, &global]( tbb::blocked_range<size_t> r ) {

// 				view& v = tls.local();

// 				for( size_t i=r.begin(); i!=r.end(); ++i ) {

// 					// “Reassign step”: Find index of centroid closest to the current point
// 					// and reassign the point to that cluster
// 					cluster_id j = reduce_min_ind(centroid, k , points[i]);

// 					// If the closest cluster is new,
// 					// change the point's cluster ID
// 					// and increment the change counter for the TL view
// 					if( j!=id[i] ) {
// 						id[i] = j; 
// 						++v.change;
// 					}

// 					// “Sum step”
// 					//		Calls the 'tally' method of the sum & count struct
// 					//		Tally adds the features of the ith point to the sum of cluster j
// 					v.array[j].tally(points[i]);
// 				}
// 			} 
// 		);
// 		// Reduce local counts to global count
// 		reduce_local_counts_to_global_count( tls, global ); 

// 	// Loop until all points are converged
// 	} while( global.change!=0 );
// }

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
std::vector<std::future<int>> futures;

for (int i = 0; i < cfg.threads; ++i) {
	futures.emplace_back(std::async(std::launch::async, do_work, std::ref(cfg)));
}

// Step 6: Wait for all threads to finish and collect exec_time_i
std::vector<int> exec_times;
for (auto& future : futures) {
	exec_times.push_back(future.get());
}

class Cluster
{
private:
	int id_cluster;
	vector<double> central_values;
	vector<Point> points;
	std::mutex c_lock; 

public:
	Cluster(int id_cluster, Point point)
	{
		this->id_cluster = id_cluster;

		int total_values = point.getTotalValues();

		// Lock the mutex before accessing shared data
        std::lock_guard<std::mutex> lock(c_lock);

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

	// Look through the vector of points
	// If you find it remove it and return true
	// If not return false
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

	// Return ID of nearest center (parallel version)
	int getIDNearestCenterParallel(const Point& point, const std::vector<Cluster>& clusters, size_t total_values) {
		double min_dist = std::numeric_limits<double>::max();
		int min_id = 0;

		// Perform parallel reduction to find the nearest center
		tbb::parallel_reduce(
			tbb::blocked_range<size_t>(0, clusters.size()),
			[&](const tbb::blocked_range<size_t>& range, double& local_min_dist, int& local_min_id) {

				for (size_t i = range.begin(); i != range.end(); ++i) {
					double sum = 0.0;

					// Calculate the squared Euclidean distance between the point and the cluster center
					for (size_t j = 0; j < total_values; ++j) {
						sum += pow(clusters[i].getCentralValue(j) - point.getValue(j), 2.0);
					}

					// Update local_min_dist and local_min_id if the calculated distance is smaller
					double dist = sqrt(sum);
					if (dist < local_min_dist) {
						local_min_dist = dist;
						local_min_id = i;
					}
				}
			},
			[&](double local_min_dist, int local_min_id) {
				// Combine results from different threads
				if (local_min_dist < min_dist) {
					min_dist = local_min_dist;
					min_id = local_min_id;
				}
			}
		);

		return min_id;
	}

	// Return ID of nearest center (uses euclidean distance)
	int getIDNearestCenter(Point point)
	{	
		// Holds the squared differences and the smallest dist found
		double sum = 0.0, min_dist;

		int id_cluster_center = 0;

		// Accumulate all the squared euclidean distances between the values and the first cluster
		// One iteration per point
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

		std::mutex clusters_mutex;
		std::mutex pointsMutex;
		//std::mutex clustersMutex;
		std::mutex doneMutex;
	}

	// Compute the K-Means algorithm
	void run(vector<Point> & points)
	{
        auto begin = chrono::high_resolution_clock::now();
        
		// Can't have more clusters than total points
		if(K > total_points){
			return;
		}

		// Holds points we have already used
		vector<int> prohibited_indexes;

		// Create initial clusters and compute their sums
		tbb::parallel_for(tbb::blocked_range<size_t>(0,K),

			//	=: captures everything by value (modifications within won't affect it)
			[=]( tbb::blocked_range<size_t> r ) {

				// Iterate over the subrange of indices assigned to the current thread
				for( size_t i=r.begin(); i!=r.end(); ++i ) {

					while(true){
						// Select random index to obtain a random point in the data set
						int index_point = rand() % total_points;

						// Check if the random point has not been selected before
						if(find(prohibited_indexes.begin(), prohibited_indexes.end(), index_point) == prohibited_indexes.end())
						{
							prohibited_indexes.push_back(index_point);

							// Set the cluster ID of the random point to the current iteration "i"
							points[index_point].setCluster(i);

							// Create a new cluster with the random point as the first member
							Cluster cluster(i, points[index_point]);

							std::lock_guard<std::mutex> lock(clusters_mutex);

							// Add our new cluster to the vector of them all
							clusters.push_back(cluster);
							break;
						}
					}
				} 
			}
		);

        auto end_phase1 = chrono::high_resolution_clock::now();
        
		int iter = 1;

		while(true)
		{
			// Tracks if any point has moved to a different cluster
			// Once this remains true, the algorithm has converged and we're done
			bool done = true;

			// Associates each point to the nearest center
			// NOTE: maybe just parallelize the getIDNearestCenter method?
			// tbb::parallel_for(tbb::blocked_range<int>(0, total_points), [&](const tbb::blocked_range<int>& range) {
			// 	for (int i = range.begin(); i != range.end(); ++i) {
			
			// 		std::lock_guard<std::mutex> lock(pointsMutex);
			// 		id_old_cluster = points[i].getCluster();
					

			// 		int id_nearest_center = getIDNearestCenter(points[i]);

			// 		if (id_old_cluster != id_nearest_center) {
			// 			if (id_old_cluster != -1) {
			// 				std::lock_guard<std::mutex> lock(clusters_mutex);
			// 				clusters[id_old_cluster].removePoint(points[i].getID());
			// 			}

			// 			std::lock_guard<std::mutex> lock(clusters_mutex);

			// 			points[i].setCluster(id_nearest_center);
			// 			clusters[id_nearest_center].addPoint(points[i]);
						
			// 			std::lock_guard<std::mutex> lock(doneMutex);
			// 			done = false;
			// 		}
			// 	}
			// });

			// Associates each point to the nearest center
			for(int i = 0; i < total_points; i++)
			{	
				// The ID of the cluster the point is currently assigned to
				int id_old_cluster = points[i].getCluster();

				// Find the closest cluster center
				int id_nearest_center = getIDNearestCenter(points[i]);

				// If the current center isn't the closest, change it
				if(id_old_cluster != id_nearest_center)
				{
					// Remove the point from its current cluster first
					if(id_old_cluster != -1){
						clusters[id_old_cluster].removePoint(points[i].getID());
					}
					// Now make its assigned cluster the new one and add it 
					points[i].setCluster(id_nearest_center);
					clusters[id_nearest_center].addPoint(points[i]);

					// Since we moved a point, convergence is not done and we must keep going
					done = false;
				}
			}

			// Recalculating the center of each cluster
			for(int i = 0; i < K; i++)
			{
				// Iterate over each value within the cluster
				for(int j = 0; j < total_values; j++)
				{	
					// Get the total number of points within the current cluster
					int total_points_cluster = clusters[i].getTotalPoints();
					double sum = 0.0;

					if(total_points_cluster > 0)
					{
						// For every point, sum up their values
						for(int p = 0; p < total_points_cluster; p++){
							sum += clusters[i].getPoint(p).getValue(j);
						}
						// Once you get the sum, set the cluster center coord to the average of all points
						clusters[i].setCentralValue(j, sum / total_points_cluster);
					}
				}
			}

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
			// Get all the points in the current cluster
			int total_points_cluster =  clusters[i].getTotalPoints();

			// Print cluster ID
			cout << "Cluster " << clusters[i].getID() + 1 << endl;

			// Loop over every point
			for(int j = 0; j < total_points_cluster; j++)
			{	
				// Print the point IDs
				cout << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";

				// Print the points' values
				for(int p = 0; p < total_values; p++){
					cout << clusters[i].getPoint(j).getValue(p) << " ";
				}
				// Get the names of the points
				string point_name = clusters[i].getPoint(j).getName();

				if(point_name != "")
					cout << "- " << point_name;

				cout << endl;
			}

			// Print value of all centers
			cout << "Cluster values: ";
			for(int j = 0; j < total_values; j++){
				cout << clusters[i].getCentralValue(j) << " ";
			}

			// Print execution time
			cout << "\n\n";
            cout << "TOTAL EXECUTION TIME = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"\n";
            
            cout << "TIME PHASE 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"\n";
            
            cout << "TIME PHASE 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-end_phase1).count()<<"\n";
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
	int num_threads = 2;
	// Initialize TBB scheduler
    tbb::task_scheduler_init init(num_threads);

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