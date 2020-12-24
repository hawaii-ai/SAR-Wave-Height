# Implements heteroskedastic regression loss functions for keras model.
import numpy as np
import tensorflow as tf

def Gaussian_NLL(y_true, y_pred, k=1, stability_factor=1e-24): 
    """                                                                                                  
    Compute Negative Log Likilihood for Gaussian Output Layer.                                           
                                                                                                         
    Args:                                                                                                
        y_true: Nxk matrix of (data) target values.                                                      
        y_pred: Nx2k matrix of parameters. Each row parameterizes                                        
                k Gaussian distributions with (mean, std).                                               
        stability_factor: Min value of sigma.                                                            
    Example:                                                                                             
        from functools import partial, update_wrapper                                                    
        loss = partial(Gaussian_NLL, k=2)                                                                
        update_wrapper(loss, Gaussian_NLL) # Keeps __name__ attribute.      
    """  
    means  = y_pred[:,:k] 
    sigmas = y_pred[:,k:] 
    sigmasafe = sigmas + stability_factor if stability_factor else sigmas  
    term1  = tf.math.log(sigmasafe) + np.log(2 * np.pi) / 2 
    term2  = tf.square((means - y_true) / sigmasafe) / 2
    nll    = term1 + term2
    nll    = tf.reduce_sum(nll, axis=-1) # Sum NLL over outputs.
    return nll 
                                                                                                         
def Gaussian_MSE(y_true, y_pred, k=1):
    """                                         
    Compute Mean Squared Error for Gaussian Output Layer.                                           
                                                                                                         
    Args:                                                                                                
        y: Nxk matrix of (data) target values.                                                           
        alpha: Nx2k matrix of parameters. Each row parameterizes                                         
               k Gaussian distributions with (mean, std).                                                
    """                                                                                                  
    means  = y_pred[:,:k] 
    rval   = tf.reduce_mean(tf.square(means - y_true), axis=-1)  
    return rval  