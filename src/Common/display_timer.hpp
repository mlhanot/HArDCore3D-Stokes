// Function to store and display wall/proc times obtained by boost::timer::cpu_timer

/*
 *
 *      This library was developed around HHO methods, although some parts of it have a more
 * general purpose. If you use this code or part of it in a scientific publication, 
 * please mention the following book as a reference for the underlying principles
 * of HHO schemes:
 *
 * The Hybrid High-Order Method for Polytopal Meshes: Design, Analysis, and Applications. 
 *  D. A. Di Pietro and J. Droniou. Modeling, Simulation and Applications, vol. 19. 
 *  Springer International Publishing, 2020, xxxi + 525p. doi: 10.1007/978-3-030-37203-3. 
 *  url: https://hal.archives-ouvertes.fr/hal-02151813.
 *
 */

#ifndef DISPLAY_TIMER_HPP
#define DISPLAY_TIMER_HPP

#include <boost/timer/timer.hpp>

namespace HArDCore3D
{

  /*!
   *	\addtogroup Common
   * @{
   */
   
    /// Function to store and display wall/proc timers from boost::timer::cpu_timer. 
    /* Wall time is in the first element of the pair, proc time in the second */
    inline std::pair<double,double> store_times(
                              boost::timer::cpu_timer & timer,  ///< The timer
                              std::string message = ""        ///< Optional message to display with time (if not message is passed, times are not displayed)
                              )
    {
      double twall = double(timer.elapsed().wall) * pow(10, -9);
      double tproc = double(timer.elapsed().user + timer.elapsed().system) * pow(10, -9);
      
      if (message != ""){
        std::cout << message << twall << "/" << tproc << std::endl;
      }

      return std::make_pair(twall, tproc);
    }

    class timer_hist {
      public:
        boost::timer::cpu_timer timer;
        std::vector<double> wtimes;
        std::vector<double> ptimes;
        std::vector<std::string> ntimes;

        timer_hist() {};

        timer_hist(bool start) {
          if (not start) timer.stop();
        }

        void start () {
          timer.start();
        }

        void stop (std::string name = "") {
          timer.stop();
          double twall = double(timer.elapsed().wall) * pow(10, -9);
          double tproc = double(timer.elapsed().user + timer.elapsed().system) * pow(10, -9);
          wtimes.emplace_back(twall);
          ptimes.emplace_back(tproc);
          ntimes.emplace_back(name);
        }

    };

  //@}

} // end of namespace HArDCore3D

#endif
