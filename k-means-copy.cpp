// This all goes after the first line of the while loop:

	// begin_first = chrono::high_resolution_clock::now();

	// Associates each point to the nearest center
	// for(int i = 0; i < total_points; i++)
	// {	
	// 	// The ID of the cluster the point is currently assigned to
	// 	int id_old_cluster = points[i].getCluster();

	// 	// Find the closest cluster center
	// 	int id_nearest_center = getIDNearestCenterPar(points[i]);

	// 	// If the current center isn't the closest, change it
	// 	if(id_old_cluster != id_nearest_center)
	// 	{
	// 		// Remove the point from its current cluster first
	// 		if(id_old_cluster != -1){
	// 			clusters[id_old_cluster].removePoint(points[i].getID());
	// 		}

	// 		// Now make its assigned cluster the new one and add it 
	// 		points[i].setCluster(id_nearest_center);
	// 		clusters[id_nearest_center].addPoint(points[i]);

	// 		// Since we moved a point, convergence is not done and we must keep going
	// 		done = false;
	// 	}
	// }

	// // end_first = chrono::high_resolution_clock::now();

	// // begin_second = chrono::high_resolution_clock::now();

	// Recalculating the center of each cluster
	// for(int i = 0; i < K; i++)
	// {
	// 	// Iterate over each value within the cluster
	// 	for(int j = 0; j < total_values; j++)
	// 	{	
	// 		// Get the total number of points within the current cluster
	// 		int total_points_cluster = clusters[i].getTotalPoints();
	// 		double sum = 0.0;

	// 		if(total_points_cluster > 0)
	// 		{
	// 			// For every point, sum up their values
	// 			for(int p = 0; p < total_points_cluster; p++){
	// 				sum += clusters[i].getPoint(p).getValue(j);
	// 			}
	// 			// Once you get the sum, set the cluster center coord to the average of all points
	// 			clusters[i].setCentralValue(j, sum / total_points_cluster);
	// 		}
	// 	}
	// }

	//end_second = chrono::high_resolution_clock::now();