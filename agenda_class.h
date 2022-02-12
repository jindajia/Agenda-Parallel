#define _CRT_SECURE_NO_DEPRECATE
#define HEAD_INFO

#include "mylib.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <map>
#include <stdlib.h>
#include <set>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include "graph.h"
#include "config.h"
#include "algo.h"
#include "query.h"
#include "build.h"
#include <sys/stat.h>

#include <boost/progress.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <chrono>


using namespace std::chrono;

using namespace boost;
using namespace boost::property_tree;

using namespace std;

class Agenda_class{

public:
    protected:
    Fwdidx fwd_idx;
    Bwdidx bwd_idx;
    ReversePushidx reverse_idx;
private:
    Graph &graph;
    double epsilon;
    double omega;
    vector<double> &inacc_idx;

//----------------voids------------------------------
public:
    Agenda_class(Graph &_graph, double _epsilon, vector<double> &_inacc_idx): graph(_graph), epsilon(_epsilon), inacc_idx(_inacc_idx)
    {
        init();
    }

    void init(){
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        omega=0;
    }

    void forward_local_update_linear_CLASS(int s,  double& rsum, double rmax, double init_residual = 1.0){

        fwd_idx.first.clean();
        fwd_idx.second.clean();

        static vector<bool> idx(graph.n);
        std::fill(idx.begin(), idx.end(), false);

        double myeps = rmax;//config.rmax;
	    if(config.with_baton == true)
		    myeps = config.beta/(config.omega*config.alpha);
		
        vector<int> q;  //nodes that can still propagate forward
        q.reserve(graph.n);
        q.push_back(-1);
        unsigned long left = 1;
        q.push_back(s);

        // residual[s] = init_residual;
        fwd_idx.second.insert(s, init_residual);
    
        idx[s] = true;
    
        while (left < (int) q.size()) {
            int v = q[left];
            idx[v] = false;
            left++;
		    //cout<<v<<endl;
            double v_residue = fwd_idx.second[v];
            fwd_idx.second[v] = 0;
            if(!fwd_idx.first.exist(v))
                fwd_idx.first.insert( v, v_residue * config.alpha);
            else
                fwd_idx.first[v] += v_residue * config.alpha;

            int out_neighbor = graph.g[v].size();
            rsum -=v_residue*config.alpha;
            if(out_neighbor == 0){
                fwd_idx.second[s] += v_residue * (1-config.alpha);
                if(graph.g[s].size()>0 && fwd_idx.second[s]/graph.g[s].size() >= myeps && idx[s] != true){
                    idx[s] = true;
                    q.push_back(s);   
                }
                continue;
            }

            double avg_push_residual = ((1.0 - config.alpha) * v_residue) / out_neighbor;
            for (int next : graph.g[v]) {
                // total_push++;
                if( !fwd_idx.second.exist(next) )
                    fwd_idx.second.insert( next,  avg_push_residual);
                else
                    fwd_idx.second[next] += avg_push_residual;

                //if a node's' current residual is small, but next time it got a laerge residual, it can still be added into forward list
                //so this is correct
                if ( fwd_idx.second[next]/graph.g[next].size() >= myeps && idx[next] != true) {  
                    idx[next] = true;//(int) q.size();
                    q.push_back(next);    
                }
            }
        }
    }
    static bool err_cmp(const pair<int,double> a,const pair<int,double> b){
	    return a.second > b.second;
    }

    void lazy_update_fwdidx_CLASS(double theta){
        if(config.no_rebuild)
        return;
	    int temp = fwd_idx.first.occur.m_num;
	    vector< pair<int,double> > error_idx(temp);
	    double rsum=0;
	    double errsum=0;
	    double inaccsum=0;
	    double OMP_start_time = omp_get_wtime();
	    for(long i=0; i<fwd_idx.first.occur.m_num; i++){
            int node_id = fwd_idx.first.occur[i];
            double reserve = fwd_idx.first[ node_id ];
	    	double residue = fwd_idx.second[ node_id ];
	    	if(residue*(1-inacc_idx[node_id])>0){
	    		//error_idx.push_back(make_pair(node_id,residue*(1-inacc_idx[node_id])));
	    		//printf("check omp i: %d\n", i);
	    		error_idx[i] = make_pair(node_id, residue*(1 - inacc_idx[node_id]));
	    		rsum+=residue;
	    		errsum+=residue*(1-inacc_idx[node_id]);
	    	}
        }
	    double OMP_end_time = omp_get_wtime();
	    printf("check OMP time-lazy-update-fwdidx: %.12f\n", OMP_end_time - OMP_start_time);
	    sort(error_idx.begin(), error_idx.end(), err_cmp);
	    long i=0;
	    double errbound=config.epsilon/graph.n*(1-theta);
	

	
	    while(errsum>errbound){
	    	update_idx(graph,error_idx[i].first);
	    	inacc_idx[error_idx[i].first]=1;
	    	errsum-=error_idx[i].second;
	    	i++;
	    }
    }


    void compute_ppr_with_fwdidx_CLASS(double check_rsum){
	
	    
        ppr.reset_zero_values();

        int node_id;
        double reserve;
	    double sum=0;
	    //INFO(fwd_idx.second.occur.m_num);
        for(long i=0; i< fwd_idx.first.occur.m_num; i++){
            node_id = fwd_idx.first.occur[i];
            reserve = fwd_idx.first[ node_id ];
            ppr[node_id] = reserve;
        }
	
        // INFO("rsum is:", check_rsum);
        if(check_rsum == 0.0)
            return;

        unsigned long long num_random_walk = config.omega*check_rsum;
        //num_total_rw += num_random_walk;

        {
            Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk
            //Timer tm(SOURCE_DIST);
            if(config.with_rw_idx){
                fwd_idx.second.occur.Sort();
                for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                    int source = fwd_idx.second.occur[i];
                    double residual = fwd_idx.second[source];
                    unsigned long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                    double a_s = residual/check_rsum*num_random_walk/num_s_rw;

                    double ppr_incre = a_s*check_rsum/num_random_walk;

                    num_total_rw += num_s_rw;
	    			rw_count += num_s_rw;

                    //for each source node, get rand walk destinations from previously generated idx or online rand walks
                    if(num_s_rw > rw_idx_info[source].second){ //if we need more destinations than that in idx, rand walk online
                        for(unsigned long k=0; k<rw_idx_info[source].second; k++){
                            int des;
                            if(config.alter_idx == 0)
                                des = rw_idx[rw_idx_info[source].first + k];
                            else
                                des = rw_idx_alter[rw_idx_info[source].first + k].back();
                            ppr[des] += ppr_incre;
                        }
                        num_hit_idx += rw_idx_info[source].second;

                        for(unsigned long j=0; j < num_s_rw-rw_idx_info[source].second; j++){ //rand walk online
                            int des = random_walk(source, graph);
                            ppr[des] += ppr_incre;
                        }
                    }else{ // using previously generated idx is enough
                        for(unsigned long k=0; k<num_s_rw; k++){
                            int des;
                            if(config.alter_idx == 0)
                                des = rw_idx[rw_idx_info[source].first + k];
                            else
                                des = rw_idx_alter[rw_idx_info[source].first + k].back();
                            ppr[des] += ppr_incre;
                        }
                        num_hit_idx += num_s_rw;
                    }
                }
            }
            else{ //rand walk online
                for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                    int source = fwd_idx.second.occur[i];
                    double residual = fwd_idx.second[source];
                    unsigned long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                    double a_s = residual/check_rsum*num_random_walk/num_s_rw;

                    double ppr_incre = a_s*check_rsum/num_random_walk;

                    num_total_rw += num_s_rw;
	    			rw_count += num_s_rw;
                    for(unsigned long j=0; j<num_s_rw; j++){
                        int des = random_walk(source, graph);
                        ppr[des] += ppr_incre;
                    }
                }
            }
        }
    }

    void Agenda_query_lazy_dynamic_CLASS(int v, double theta){
        double rsum = 1.0;
	    double temp_eps=epsilon;
        epsilon=config.epsilon*theta;
	    omega = (2+config.epsilon)*log(2/config.pfail)/config.delta/epsilon/epsilon;
        forward_local_update_linear_CLASS(v, rsum, config.rmax);
        epsilon=temp_eps;

        double OMP_part1_start=omp_get_wtime();
		lazy_update_fwdidx_CLASS(theta);
		double OMP_part1_end=omp_get_wtime();
        printf("OMP check part1: %.12f\n", OMP_part1_end-OMP_part1_start);

        double OMP_part2_start=omp_get_wtime();
        compute_ppr_with_fwdidx_CLASS(rsum);
        double OMP_part2_end=omp_get_wtime();
        printf("OMP check part2: %.12f\n", OMP_part2_end-OMP_part2_start);
    }

    void Agenda_update(){

    }

};
