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
        10.0 - 0.2*y - 0.27 * y.powf(1.5)
    };
    /* Create an ARC (Automatic Reference Counted Shared pointer) to the Closure that allows
       us to 'share' the Function between the different threads without worrying about 
       thread safety or other shared resources. */
    let de_share = Arc::new(de);

    let time = time::precise_time_s();
    
    //Before each thread, create a copy of the Arc pointer
    let euler_copy = de_share.clone();

    /* For the closure, move ownership of the parameters to the thread (the copy of the
       pointer in particulare is key here! */
    let answer = thread::spawn(move || {
        euler_method(0.0,(0.0,5.0),0.0000001, euler_copy)
    });

    let heun_copy = de_share.clone();
    let imp_answer = thread::spawn(move || {
        improved_euler(0.0,(0.0,5.0), 0.0000001, heun_copy)
    });
    
    let runge_copy = de_share.clone();
    let runge_answer = thread::spawn(move || {
        runge_kutta(0.0,(0.0,5.0),0.0000001, runge_copy)
    });

    /*Ensure the Main thread does not exit until the Approximation threads
    are all complete so that the results print. */
    let a = answer.join().unwrap().1;
    let b = imp_answer.join().unwrap().1;
    let c = runge_answer.join().unwrap().1;
    let linear_time = a + b + c;
    let total = time::precise_time_s() - time;

    println!("---RESULTS---");
    println!("Total Time: {}", total);
    println!("Linear Time: {}", linear_time);
    println!("Time Saved: {}\n", linear_time - total); 
}


fn euler_method<F:Fn(f64,f64)->f64>(ystart:f64,trange:(f64,f64),h:f64,de:Arc<F>)-> (f64,f64) {
    let start = time::precise_time_s();
    let mut ycurrent = ystart;
    let mut tcurrent = trange.0;
    while tcurrent < trange.1 {
        ycurrent += h*de(tcurrent,ycurrent);
        tcurrent += h;
    }
    let end = time::precise_time_s() - start;
    println!("Euler Approximation: {}\nTime: {}\n", ycurrent, end);
    (ycurrent,end)
}

/* Also known as the Heun method for approximation*/
fn improved_euler<F:Fn(f64,f64)->f64>(ystart:f64,trange:(f64,f64),h:f64,de:Arc<F>) -> (f64,f64) {
    let start = time::precise_time_s();
    let mut ycurrent = ystart;
    let mut tcurrent = trange.0;
    let half_h = h/2.0;
    while tcurrent < trange.1 {
        let ynext = ycurrent + h*de(tcurrent,ycurrent);
        ycurrent += half_h*(de(tcurrent,ycurrent) + de(tcurrent + h,ynext));
        tcurrent += h;
    }
    let end = time::precise_time_s() - start;
    println!("Heun Approximation: {}\nTime: {}\n", ycurrent, end);
    (ycurrent,end)
}

fn runge_kutta<F:Fn(f64,f64)->f64>(ystart:f64,trange:(f64,f64),h:f64,de:Arc<F>)-> (f64,f64) {
    let start = time::precise_time_s();
    let mut ycurrent = ystart;
    let mut tcurrent = trange.0;

    /* Precompute division operations outside of loop for efficiency*/
    let half_h = h/2.0;
    let sixth_h = h/6.0;
    while tcurrent < trange.1 {
        let k1 = de(tcurrent,ycurrent);
        let k2 = de(tcurrent + half_h, ycurrent + half_h*k1);
        let k3 = de(tcurrent + half_h, ycurrent + half_h*k2);
        let k4 = de(tcurrent + h, ycurrent + h*k3);
        ycurrent += sixth_h*(k1 + 2.0*k2 + 2.0*k3 + k4);
        tcurrent += h;
    }
    let end = time::precise_time_s() - start;
    println!("Runge Kutta Approximation: {}\nTime: {}\n",ycurrent, end);
    (ycurrent,end)
}
