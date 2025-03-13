#include "SemidiscreteOT.h"
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

int main(int argc, char* argv[]) {
    try {
        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
        MPI_Comm mpi_communicator = MPI_COMM_WORLD;
        const unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

        // Create conditional output stream
        dealii::ConditionalOStream pcout(std::cout, this_mpi_process == 0);
        
        SemidiscreteOT<3> semidiscrete_ot(mpi_communicator);
        
        // Use command line argument if provided, otherwise use default
        std::string param_file = (argc > 1) ? argv[1] : "parameters.prm";
        
        pcout << "Using parameter file: " << param_file << std::endl;
        dealii::ParameterAcceptor::initialize(param_file);
        
        semidiscrete_ot.run();
    }
    catch(std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
    return 0;
}