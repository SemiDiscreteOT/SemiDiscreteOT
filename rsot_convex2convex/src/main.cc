#include "rsot.h"

int main(int argc, char* argv[]) {
    try {
        Convex2Convex<3> convex2convex;
        
        // Use command line argument if provided, otherwise use default
        std::string param_file = (argc > 1) ? argv[1] : "parameters.prm";
        
        std::cout << "Using parameter file: " << param_file << std::endl;
        ParameterAcceptor::initialize(param_file);
        
        convex2convex.run();
    }
    catch(std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
    return 0;
}