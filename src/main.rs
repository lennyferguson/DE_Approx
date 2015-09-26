//! Author: Stewart ChOAarles

use std::thread;
use std::sync::Arc;
extern crate time;

#[allow(unused_must_use)]
#[allow(unused_variables)]
fn main() {
    /* This closure represents the DE that we are approximating. This particular example does not
       actually use the 't' parameter, though there are DE's that do have the 't' parameter.
       The different approximation techniques utilize the 't' parameter in their general 'solution' 
       to the approximation, which makes it necessary to include for our 'de'.*/
    let de = |t:f64,y:f64| {
        10.0 - 0.2 * y - 0.27 * y.powf(1.5)
    };
    /* Create an ARC (Automatic Reference Counted Shared pointer) to the Closure that allows
       us to 'share' the Function between the different threads without worrying about 
       thread safety or other shared resources. */
    let de_share = Arc::new(de);
    
    println!("\n----Threaded Execution with Atomic Locks----\n");

    const Y0:f64 = 0.0;
    const T0:f64 = 0.0;
    const T_END:f64 = 5.0;
    const H:f64 = 0.00000001;
    let steps = ((T_END - T0)/H).round() as usize;
    
    let time = time::precise_time_s();

       
    /* For the closure, move ownership of the parameters to the thread (the copy of the
    pointer in particulare is key here!
    */
    let euler_copy = de_share.clone();
    let answer = thread::spawn(move || {
        threaded_funcs::euler_method(Y0, T0, steps, H, euler_copy)
    });

    let heun_copy = de_share.clone();
    let imp_answer = thread::spawn(move || {
        threaded_funcs::improved_euler(Y0, T0, steps, H, heun_copy)
    });
    
    let runge_copy = de_share.clone();
    let runge_answer = thread::spawn(move || {
        threaded_funcs::runge_kutta(Y0, T0, steps, H, runge_copy)
    });
    
    /*Ensure the Main thread does not exit until the Approximation threads
    are all complete so that the results print. 
     
     The order of these does not actually matter. The total time will be approximately
     equal to the time it takes the longest running thread to join. Similarly, the serialized time
     will be a function of the length of individual times. 
     The order of the joins has no significant meaning.*/
    answer.join();
    imp_answer.join();
    runge_answer.join();

    let total = time::precise_time_s() - time;

    println!("----Serialized Execution without Atomics----\n");
    
    let mut serialized = 0.0;
    serialized += serial_funcs::euler_method(Y0,T0,steps,H,&serial_funcs::regular_de).1;
    serialized += serial_funcs::improved_euler(Y0,T0,steps,H,&serial_funcs::regular_de).1;
    serialized += serial_funcs::runge_kutta(Y0,T0,steps,H,&serial_funcs::regular_de).1;

    println!("---RESULTS---");
    println!("Threaded Time: {} seconds", total);
    println!("Serialized Time: {} seconds", serialized);
    println!("Gains from Threading: {} seconds\n", serialized - total); 
}

mod threaded_funcs {
    use std::sync::Arc;
    extern crate time;
    
    #[allow(unused_variables)]
    pub fn euler_method<F:Fn(f64,f64)->f64>(y0:f64, t0:f64, steps:usize, h:f64,de:Arc<F>)-> (f64,f64) {
        let start = time::precise_time_s();
        let mut ycurrent = y0;
        let mut tcurrent = t0;
        for x in 0..steps {
            ycurrent += h*de(tcurrent,ycurrent);
            tcurrent += h;
        }
        let end = time::precise_time_s() - start;
        println!("Euler Approximation: {}\nTime: {} seconds\n", ycurrent, end);
        (ycurrent,end)
    }
    
    #[allow(unused_variables)]
    /* Also known as the Heun method for approximation*/
    pub fn improved_euler<F:Fn(f64,f64)->f64>(y0:f64, t0:f64, steps:usize, h:f64, de:Arc<F>) -> (f64,f64) {
        let start = time::precise_time_s();
        let mut ycurrent = y0;
        let mut tcurrent = t0;
        let half_h = h/2.0;
        for x in 0..steps {
            let k1 = de(tcurrent,ycurrent);
            ycurrent += half_h*(k1 + de(tcurrent + h , ycurrent + h*k1));
            tcurrent += h;
        }
        let end = time::precise_time_s() - start;
        println!("Heun Approximation: {}\nTime: {} seconds\n", ycurrent, end);
        (ycurrent,end)
    }
    
    #[allow(unused_variables)]
    pub fn runge_kutta<F:Fn(f64,f64)->f64>(y0:f64, t0:f64, steps:usize, h:f64, de:Arc<F>)-> (f64,f64) {
        let start = time::precise_time_s();
        let mut ycurrent = y0;
        let mut tcurrent = t0;
        
        /* Precompute division operations outside of loop for efficiency*/
        let half_h = h/2.0;
        let sixth_h = h/6.0;
        for x in 0..steps {
            let k1 = de(tcurrent,ycurrent);
            let k2 = de(tcurrent + half_h, ycurrent + half_h*k1);
            let k3 = de(tcurrent + half_h, ycurrent + half_h*k2);
            let k4 = de(tcurrent + h, ycurrent + h*k3);
            ycurrent += sixth_h*(k1 + 2.0*k2 + 2.0*k3 + k4);
            tcurrent += h;
        }
        let end = time::precise_time_s() - start;
        println!("Runge Kutta Approximation: {}\nTime: {} seconds\n",ycurrent, end);
        (ycurrent,end)
    }
}

mod serial_funcs {
    extern crate time;

    #[allow(unused_variables)]
    pub fn regular_de(t:f64,y:f64)->f64 {
        10.0 - 0.2 * y - 0.27 * y.powf(1.5)
    }
    
    #[allow(unused_variables)]
    pub fn euler_method<F:Fn(f64,f64)->f64>(y0:f64,t0:f64,steps:usize, h:f64,de:&F)-> (f64,f64) {
        let start = time::precise_time_s();
        let mut ycurrent = y0;
        let mut tcurrent = t0;
        for x in 0..steps {
            ycurrent += h*de(tcurrent,ycurrent);
            tcurrent += h;
        }
        let end = time::precise_time_s() - start;
        println!("Euler Approximation: {}\nTime: {} seconds\n", ycurrent, end);
        (ycurrent,end)
    }
    
    #[allow(unused_variables)]
    /* Also known as the Heun method for approximation*/
    pub fn improved_euler<F:Fn(f64,f64)->f64>(y0:f64,t0:f64,steps:usize, h:f64,de:&F)-> (f64,f64) {
        let start = time::precise_time_s();
        let mut ycurrent = y0;
        let mut tcurrent = t0;
        let half_h = h/2.0;
        for x in 0..steps {
            let k1 = de(tcurrent,ycurrent);
            ycurrent += half_h*(k1 + de(tcurrent + h , ycurrent + h*k1));
            tcurrent += h;
        }
        let end = time::precise_time_s() - start;
        println!("Heun Approximation: {}\nTime: {} seconds\n", ycurrent, end);
        (ycurrent,end)
    }
    
    #[allow(unused_variables)]
    pub fn runge_kutta<F:Fn(f64,f64)->f64>(y0:f64,t0:f64,steps:usize, h:f64,de:&F)-> (f64,f64) {
        let start = time::precise_time_s();
        let mut ycurrent = y0;
        let mut tcurrent = t0;
        
        /* Precompute division operations outside of loop for efficiency*/
        let half_h = h/2.0;
        let sixth_h = h/6.0;
        for x in 0..steps {
            let k1 = de(tcurrent,ycurrent);
            let k2 = de(tcurrent + half_h, ycurrent + half_h*k1);
            let k3 = de(tcurrent + half_h, ycurrent + half_h*k2);
            let k4 = de(tcurrent + h, ycurrent + h*k3);
            ycurrent += sixth_h*(k1 + 2.0*k2 + 2.0*k3 + k4);
            tcurrent += h;
        }
        let end = time::precise_time_s() - start;
        println!("Runge Kutta Approximation: {}\nTime: {} seconds\n",ycurrent, end);
        (ycurrent,end)
    }
}
