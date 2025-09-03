#include "application.h"

int main() {
    application app;
    if (app.initialize()) {
        app.run();
    }
    app.shutdown();
    return 0;
}
