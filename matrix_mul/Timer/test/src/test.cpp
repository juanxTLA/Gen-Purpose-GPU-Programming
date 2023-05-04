#include <Timer.hpp>
#include <unistd.h>
#include <iostream>

int main(int argc, char** argv){

    std::cout << "Starting timer test - Timing on local machine" << std::endl;
    
    Timer timer1;
    timer1.start();

    sleep(5);

    timer1.stop();

    Timer timer2;
    timer2.start();

    sleep(1);

    timer2.stop();

    std::cout << "\tTIMER1 AFTER 5s SLEEP - " << timer1.elapsedTime_ms() << std::endl;
    std::cout << "\tTIMER2 AFTER 1s SLEEP - " << timer2.elapsedTime_ms() << std::endl;

    std::cout << "Testing Resolution" << std::endl;

    double runningSum = 0.0f;
    int runs = 1000;

    for(int i = 0; i < runs; ++i){
        Timer timer3;
        timer3.start();
        timer3.stop();

        runningSum += timer3.elapsedTime_ms();
    }
    std::cout << "\tResolution: " << runningSum/runs <<  std::endl;

    return 0;

}