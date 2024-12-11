#include "rsot.h"

int main() {
    try {
        Convex2Convex<3> convex2convex;
        ParameterAcceptor::initialize("parameters.prm");
        convex2convex.run();
    }
    catch(std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
    return 0;
}