
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


string get_time_path() {
    using namespace boost::posix_time;
    auto tm = second_clock::local_time();
#ifdef WIN32
    return  "../../execution/" + to_iso_string(tm);
#else
    return parent_folder+FILESEP+"execution/" + to_iso_string(tm);
#endif
}

#include <boost/program_options.hpp>

namespace po = boost::program_options;


using namespace std;

int main(int argc, char *argv[]) {
    ios::sync_with_stdio(false);
    program_start(argc, argv);

    // this is main entry
    Saver::init();
    srand(time(NULL));
    config.graph_alias = "nethept";
    for (int i = 0; i < argc; i++) {
        string help_str = ""
                "fora query --algo <algo> [options]\n"
                "fora topk  --algo <algo> [options]\n"
                "fora batch-topk --algo <algo> [options]\n"
                "fora build [options]\n"
                "fora generate-ss-query [options]\n"
                "fora gen-exact-topk [options]\n"
                "fora\n"
                "\n"
                "algo: \n"
                "  bippr\n"
                "  montecarlo\n"
                "  fora\n"
                "  hubppr\n"
                "options: \n"
                "  --prefix <prefix>\n"
                "  --epsilon <epsilon>\n"
                "  --dataset <dataset>\n"
                "  --query_size <queries count>\n"
                "  --k <top k>\n"
                "  --with_idx\n"
                "  --hub_space <hubppr oracle space-consumption>\n"
                "  --exact_ppr_path <eaact-topk-pprs-path>\n"
                "  --rw_ratio <rand-walk cost ratio>\n"
                "  --result_dir <directory to place results>"
                "  --rmax_scale <scale of rmax>\n";

        if (string(argv[i]) == "--help") {
            cout << help_str << endl;
            exit(0);
        }
    }

    config.action = argv[1];
    cout << "action: " << config.action << endl;

    // init graph first
    for (int i = 0; i < argc; i++) {
        if (string(argv[i]) == "--prefix") {
            config.prefix = argv[i + 1];
        }
        if (string(argv[i]) == "--dataset") {
            config.graph_alias = argv[i + 1];
        }
    }

    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--algo") {
    		config.algo = string(argv[i + 1]);
        }else if (arg == "--epsilon") {
            config.epsilon = atof(argv[i + 1]);
            INFO(config.epsilon);
        }else if(arg == "--multithread"){
             config.multithread = true;
        }
        else if(arg == "--result_dir"){
            config.exe_result_dir = string(argv[i + 1]);
        }
        else if( arg == "--exact_ppr_path"){
            config.exact_pprs_folder = string(argv[i + 1]);
        }
        else if(arg == "--with_idx"){
            config.with_rw_idx = true;
        }
        else if(arg == "--rmax_scale"){
            config.rmax_scale = atof(argv[i+1]);
        }
        else if (arg == "--query_size"){
            config.query_size = atoi(argv[i+1]);
        }
        else if (arg == "--update_size"){
            config.update_size = atoi(argv[i+1]);
        }
        else if (arg == "--check_size"){
            config.check_size = atoi(argv[i+1]);
        }
        else if (arg == "--check_from"){
            config.check_from = atoi(argv[i+1]);
        }
        else if(arg == "--hub_space"){
            config.hub_space_consum = atoi(argv[i+1]);
        }
        else if (arg == "--version"){
            config.version = argv[i+1];
        }
        else if(arg == "--k"){
            config.k = atoi(argv[i+1]);
        }
        else if(arg == "--rw_ratio"){
            config.rw_cost_ratio = atof(argv[i + 1]);
        }
        else if (arg == "--prefix" || arg == "--dataset") {
            // pass
        }
		else if (arg == "--baton"){
            config.with_baton = true;
        }else if (arg == "--exact"){
			config.exact = true;
		}
		else if (arg == "--reuse"){
            config.reuse = true;
        }
		else if (arg == "--beta"){
            config.beta = atof(argv[i + 1]);
        }
		else if (arg == "--n"){
			config.n = atof(argv[i + 1]);
		}
		else if (arg == "--power_iter"){
			config.power_iteration=true;
		}
		else if (arg == "--adapt"){
			config.adaptive=true;
		}
		else if (arg == "--alter_idx"){
			config.alter_idx=true;
		}
        else if(arg == "--opt"){
            config.opt = true;
        }
        else if(arg == "--balanced"){
            config.balanced = true;
        }
        else if(arg == "--errorlimiter"){
            config.errorlimiter = atof(argv[i + 1]);
        }
        else if(arg == "--with_fora"){
            config.with_fora = true;
        }
        else if(arg == "--no_rebuild"){
            config.no_rebuild = true;
        }
        else if(arg == "--graph_n"){
            config.graph_n = atoi(argv[i+1]);
        }
        else if(arg == "--insert_ratio"){
            config.insert_ratio = atof(argv[i + 1]);
        }
        else if (arg.substr(0, 2) == "--") {
            cerr << "command not recognize " << arg << endl;
            exit(1);
        }
    }

    INFO(config.version);
    vector<string> possibleAlgo = {BIPPR, FORA, BATON, FWDPUSH, MC, HUBPPR, PARTUP, LAZYUP, GENDA, RESACC};
	if(config.algo==BATON||config.algo==PARTUP||config.algo==LAZYUP||config.algo==GENDA){
		config.with_baton = true;
	}	

    if(config.with_fora){
        config.with_baton = false;
    }
    
    INFO(config.action);

    srand (time(NULL));
    if (config.action == QUERY) {
        auto f = find(possibleAlgo.begin(), possibleAlgo.end(), config.algo);
        assert (f != possibleAlgo.end());
        if(f == possibleAlgo.end()){
            INFO("Wrong algo param: ", config.algo);
            exit(1);
        }

        config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO("load graph finish");
        init_parameter(config, graph);
   
        INFO("finihsed initing parameters");
        INFO(graph.n, graph.m);

        // if(config.multithread){
        //     init_multi_setting(graph.n);
        // }

        if(config.with_rw_idx)
            deserialize_idx();
        INFO(rw_idx.size());
        query(graph);
		
		//cout <<  Timer::used(RONDOM_WALK)*1000/double(rw_count) << "ms" << " for each random walk" <<endl;
		//cout <<  Timer::used(FWD_LU)*1000/double(fp_count) << "ms" << " for each forward push" <<endl;
		
    }
    else if (config.action == GEN_SS_QUERY){
        config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO(graph.n, graph.m);

        generate_ss_query(graph.n);
    }
	else if (config.action == GEN_UPDATE){
        config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO(graph.n, graph.m);

        generate_update(graph);
    }
    else if (config.action == TOPK){
        auto f = find(possibleAlgo.begin(), possibleAlgo.end(), config.algo);
        assert (f != possibleAlgo.end());
        if(f == possibleAlgo.end()){
            INFO("Wrong algo param: ", config.algo);
            exit(1);
        }
        
        config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO("load graph finish");
        init_parameter(config, graph);
   
        if(config.exact_pprs_folder=="" || !exists_test(config.exact_pprs_folder))
            config.exact_pprs_folder = config.graph_location;

        INFO("finihsed initing parameters");
        INFO(graph.n, graph.m);

        // if(config.multithread){
        //     init_multi_setting(graph.n);
        // }

        if(config.with_rw_idx)
            deserialize_idx();
            
        topk(graph);
		cout<<"Cycle: "<<cycle<<endl;
		cout<<"Randomwalk: "<<rw_count<<endl;
		cout<<"Boundupdate: "<<bound_update_count<<endl;
    }
    else if(config.action == BATCH_TOPK){
        auto f = find(possibleAlgo.begin(), possibleAlgo.end(), config.algo);
        assert(f != possibleAlgo.end());
        if(f == possibleAlgo.end()){
            INFO("Wrong algo param: ", config.algo);
            exit(1);
        }
        
        load_exact_topk_ppr();

        config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO("load graph finish");
        init_parameter(config, graph);
   
        if(config.exact_pprs_folder=="" || !exists_test(config.exact_pprs_folder))
            config.exact_pprs_folder = config.graph_location;

        INFO("finihsed initing parameters");
        INFO(graph.n, graph.m);

        if(config.with_rw_idx)
            deserialize_idx();
            
        batch_topk(graph);
    }
    else if (config.action == GEN_EXACT_TOPK){
        config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO("load graph finish");
        init_parameter(config, graph);

        if(config.exact_pprs_folder=="" || !exists_test(config.exact_pprs_folder))
            config.exact_pprs_folder = config.graph_location;
   
        INFO("finihsed initing parameters");
        INFO(graph.n, graph.m);

        gen_exact_topk(graph);
    }
    else if(config.action == BUILD){
        config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO("load graph finish");
        init_parameter(config, graph);
   
        INFO("finihsed initing parameters");
        INFO(graph.n, graph.m);

        Timer tm(0);
        if(config.multithread)
            multi_build(graph);
        else{
            if(config.opt){
                build_alter(graph);
            }
            else if(config.alter_idx){
                build_vldb2010(graph);
            }else{
                build(graph);
            }
        }
        
    }
	else if(config.action == DYNAMIC_SS){
		double OMP_total_start_time=omp_get_wtime();
		config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO("load graph finish");
        init_parameter(config, graph);
		
		std::string prefix = "result/"+config.graph_alias;
		INFO(access(prefix.c_str(), NULL));
		if (access(prefix.c_str(), NULL) == -1)
			mkdir(prefix.c_str(),0777);	
   
        INFO("finished initing parameters");
        INFO(graph.n, graph.m);
		
		if(config.with_rw_idx&&!config.exact)
            deserialize_idx();
		
        cout<<"deserialize completed"<<endl;
		int total_threads = omp_get_num_threads();
		printf("total_cores= %d\n", total_threads);
		generate_dynamic_workload();
		
		if(config.adaptive){
			set_optimal_beta(config,graph);
			rebuild_idx(graph);
		}
        omp_set_num_threads(2);
		dynamic_ssquery(graph);
		double OMP_total_end_time=omp_get_wtime();
		printf("OMP CHECK TOTAL TIME%.12f\n", OMP_total_end_time-OMP_total_start_time);
		cout << "Memory usage (MB):" << get_proc_memory()/1000.0 << endl << endl; 
    }
    else if(config.action == DYNAMIC_SS_PARALLEL){//------------------PARALLEL-----------!!----------------------------------
		config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO("load graph finish");
        init_parameter(config, graph);
		Graph graph_2(graph);

		std::string prefix = "result/"+config.graph_alias;
		INFO(access(prefix.c_str(), NULL));
		if (access(prefix.c_str(), NULL) == -1)
			mkdir(prefix.c_str(),0777);	
   
        INFO("finished initing parameters");
        INFO(graph.n, graph.m);

		cout<<"Please input the number of worker threads"<<endl;
		int num_total_worker;
        cin>>num_total_worker;

		if(config.with_rw_idx&&!config.exact)
            deserialize_idx_forParallel(num_total_worker);
		
        cout<<"deserialize completed"<<endl;
		
		generate_parallel_dynamic_workload_workspace();
		
        cout<<"Generate workload&workspace completed"<<endl;
		if(config.adaptive){
			set_optimal_beta(config,graph);
			rebuild_idx(graph);
		}

        double OMP_total_start_time=omp_get_wtime();
        dynamic_ssquery_parallel(graph, graph_2, num_total_worker);
		double OMP_total_end_time=omp_get_wtime();
		printf("OMP CHECK TOTAL TIME%.12f\n", OMP_total_end_time-OMP_total_start_time);
		cout << "Memory usage (MB):" << get_proc_memory()/1000.0 << endl << endl; 
        
    }
	else if(config.action == DYNAMIC_TOPK){
		config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO("load graph finish");
        init_parameter(config, graph);
		
		std::string prefix = "result/"+config.graph_alias;
		INFO(access(prefix.c_str(), NULL));
		if (access(prefix.c_str(), NULL) == -1)
			mkdir(prefix.c_str(),0777);	
	
        INFO("finished initing parameters");
        INFO(graph.n, graph.m);
		
		deserialize_idx();
		
		generate_dynamic_workload();
        
		dynamic_topk(graph);
		
	}
	else if(config.action == DYNAMIC_ONEHOP){
		config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO("load graph finish");
        init_parameter(config, graph);
		if(config.adaptive){
			set_optimal_beta_onehop(config,graph);
			rebuild_idx(graph);
		}
	
        INFO("finished initing parameters");
        INFO(graph.n, graph.m);
		
		if(config.algo!="BATON")
			deserialize_idx();
		
		generate_dynamic_workload();
        
		dynamic_onehop(graph);
		cout<<"Randomwalk: "<<rw_count<<endl;
		cout<<"Hit: "<<num_hit_idx<<endl;
		
	}
	else if(config.action == DYNAMIC_HYBRID){
		config.graph_location = config.get_graph_folder();
        Graph graph(config.graph_location);
        INFO("load graph finish");
        init_parameter(config, graph);
	
        INFO("finished initing parameters");
        INFO(graph.n, graph.m);
		
		if(config.algo==FORA_AND_BATON)
			deserialize_idx_all();
		else
			deserialize_idx();
		
		generate_dynamic_workload(true);
        
		dynamic_hybrid(graph);
		
	}
	else if(config.action == CALCULATE_ACCURACY){
		//vector<string> algos = {FORA, BATON, FORA_NO_INDEX, PARTUP, LAZYUP, RESACC};
        vector<string> algos = {BATON};
		vector<vector<PPR_Result>> exact_ppr_result;
		std::string exact_path = "result/"+config.graph_alias+"/"+"exact"+".txt";
		
		std::string result_path = "result/"+config.graph_alias+"/"+"C_accuracy"+".txt";
		std::string top_result_path = "result/"+config.graph_alias+"/"+"C_accuracy_topk"+".txt";
		ofstream result_file, top_result_file;
		result_file.open(result_path);
		top_result_file.open(top_result_path);
		config.graph_location = config.get_graph_folder();
		Graph graph(config.graph_location);
		if(access(exact_path.c_str(), NULL)==-1)
			exit(1);
		else
			load_ppr_result(exact_ppr_result, exact_path);		
		for (auto algo : algos){
			std::string prefix = "result/"+config.graph_alias+"/"+algo+".txt";
			result_file<<algo<<endl;
			if(access(prefix.c_str(), NULL)==-1)
				continue;
			vector<vector<PPR_Result>> algo_ppr_result;
			load_ppr_result(algo_ppr_result, prefix);
			INFO(algo);
			calc_accuracy(algo_ppr_result, exact_ppr_result, result_file, graph.n);
			result_file<<"end"<<endl;
		}
        /*
		for (auto algo : algos){
			std::string prefix = "result/"+config.graph_alias+"/"+algo+"_topk.txt";
			result_file<<algo<<"_topk"<<endl;
			if(access(prefix.c_str(), NULL)==-1)
				continue;
			vector<vector<PPR_Result>> algo_ppr_result;
			load_ppr_result(algo_ppr_result, prefix);
			INFO(algo);
			calc_accuracy(algo_ppr_result, exact_ppr_result, result_file, config.k);
			result_file<<"end"<<endl;
		}*/
		/*
		for (auto algo : algos){
			top_result_file<<algo<<endl;
			for(int i=0;i<6;i++){
			
				std::string prefix = "result/"+config.graph_alias+"/"+algo+".txt";
				
				if(access(prefix.c_str(), NULL)==-1)
					continue;
				vector<vector<PPR_Result>> algo_ppr_result;
				load_ppr_result(algo_ppr_result, prefix);
				INFO(algo);
				INFO(pow(10,i));
				calc_accuracy(algo_ppr_result, exact_ppr_result, top_result_file, pow(10,i));
			}
			top_result_file<<"end"<<endl;
		}
        */
	}
    else {
        cerr << "sub command not regoznized" << endl;
        exit(1);
    }

    Timer::show();
	INFO(config.beta);
	
    if(config.action == QUERY || config.action == TOPK || config.action == DYNAMIC_SS){
        Counter::show();
        auto args = combine_args(argc, argv);
        Saver::save_json(config, result, args);
    }
    else

    program_stop();
    return 0;
}
