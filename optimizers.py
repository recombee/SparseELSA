#NAdamS

import keras

from keras import backend
from keras import ops
from keras.src.backend.torch.core import *

class NadamS(keras.optimizers.Optimizer):
    """Optimizer that implements the Nadam algorithm.
    
    Much like Adam is essentially RMSprop with momentum, Nadam is Adam with
    Nesterov momentum.
    
    Updates only for nonzero gradients or only for vars specified by slicer.

    Args:
        learning_rate: A float, a
            `keras_core.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.001`.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates.
            Defaults to `0.9`.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 2nd moment estimates. Defaults to
            `0.999`.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
            Defaults to `1e-7`.
        {{base_optimizer_keyword_args}}

    Reference:

    - [Dozat, 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).

    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        name="nadam",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            **kwargs,
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def build(self, var_list):
        """Initialize optimizer variables.

        Nadam optimizer has 2 types of variables: momentums and velocities.

        Args:
            var_list: list of model variables to build Nadam variables on.
        """
        if self.built:
            return
        if var_list:
            dtype = var_list[0].dtype
        else:
            dtype = backend.floatx()
        super().build(var_list)
        self._momentums = []
        self._velocities = []
        self._u_product = keras.Variable(1.0, dtype=dtype)

        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="momentum"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="velocity"
                )
            )
    
    def apply_gradients(self, grads_and_vars, slicer=None):
        grads, trainable_variables = zip(*grads_and_vars)
        self.apply(grads, trainable_variables, slicer)
        # Return iterations for compat with tf.keras.
        return self.iterations

    def apply(self, grads, trainable_variables=None, slicer=None):
        """
        `grads` should be a list of gradient tensors
        with 1:1 mapping to the list of variables the optimizer was built with.

        `variables` can be provided on the first call to build the optimizer.
        """
        if len(grads) == 0:
            # It is possible that the grad is empty. In this case,
            # `apply_gradients` is a no-op.
            return

        if trainable_variables is None:
            if not self.built:
                raise ValueError(
                    "When passing `grads` without `variables`, the optimizer "
                    "must already be built on a list of variables. "
                    "Call `optimizer.build(trainable_variables)` first. "
                )
            if len(grads) != len(self._trainable_variables_indices):
                raise ValueError(
                    "When passing `grads` as a list of gradient tensors, the "
                    f"gradients must match `optimizer.variables` one-to-on. "
                    f"Received a list of {len(grads)} gradients, but the "
                    f"optimizer is tracking {len(self._trainable_variables)} "
                    "trainable variables."
                )
            trainable_variables = self._trainable_variables
        else:
            trainable_variables = list(trainable_variables)
            # Optionally build optimizer.
            if not self.built:
                with keras.src.backend.name_scope(self.name, caller=self):
                    self.build(trainable_variables)
                self.built = True
            self._check_variables_are_known(trainable_variables)

        with keras.src.backend.name_scope(self.name, caller=self):
            # Filter empty gradients.
            grads, trainable_variables = self._filter_empty_gradients(
                grads, trainable_variables
            )
            if len(list(grads)) == 0:
                return

            # Unscale gradients.
            #scale = self.loss_scale_factor
            #if scale is not None:
            #    grads = [g if g is None else g / scale for g in grads]

            # Apply clipping and weight decay.
            grads = self._clip_gradients(grads)
            self._apply_weight_decay(trainable_variables)

            # Apply gradient updates.
            self._internal_apply_gradients(
                list(zip(grads, trainable_variables)), slicer
            )

            # Apply variable constraints after applying gradients.
            for variable in trainable_variables:
                if getattr(variable, "constraint", None) is not None:
                    variable.assign(variable.constraint(variable))
    
    def _internal_apply_gradients(self, grads_and_vars, slicer=None):
        dtype = self._u_product.dtype
        self._u_product.assign(
            self._u_product
            * self.beta_1
            * (
                1.0
                - 0.5 * ops.power(0.96, ops.cast(self.iterations + 1, dtype))
            )
        )
        
        for grad, var in grads_and_vars:
            self.update_step(grad, var, self.learning_rate, slicer)
        
        self.iterations.assign(self.iterations + 1)
    
    def update_step(self, gradient, variable, learning_rate, slicer=None):
        """Update step given gradient and the associated model variable."""
        var_dtype = variable.dtype
        lr = ops.cast(learning_rate, var_dtype)
        gradient = ops.cast(gradient, var_dtype)
#        print(gradient)
#        print(gradient.shape)
#        print(torch.count_nonzero(gradient, dim=1))
#        print("1",torch.where(torch.count_nonzero(gradient, dim=1)!=0))
        
#        print("grad",gradient[torch.where(torch.count_nonzero(gradient, dim=1)!=0)].shape)
#        print("var",variable[torch.where(torch.count_nonzero(gradient, dim=1)!=0)].shape)
  
        if slicer is None:
            slicer = torch.where(torch.count_nonzero(gradient, dim=1)!=0)
        else:
            slicer = (slicer, )
        
        
        
        #arg_slicer = torch.argwhere(torch.count_nonzero(gradient, dim=1)!=0)
        nonzero_gradient = gradient[slicer]
        nonzero_variable = variable[slicer]
        
        local_step = ops.cast(self.iterations + 1, var_dtype)
        next_step = ops.cast(self.iterations + 2, var_dtype)
        decay = ops.cast(0.96, var_dtype)
        beta_1 = ops.cast(self.beta_1, var_dtype)
        beta_2 = ops.cast(self.beta_2, var_dtype)
        
        u_t = beta_1 * (1.0 - 0.5 * (ops.power(decay, local_step)))
        u_t_1 = beta_1 * (1.0 - 0.5 * (ops.power(decay, next_step)))
        u_product_t = ops.cast(self._u_product, var_dtype)

        u_product_t_1 = u_product_t * u_t_1
        beta_2_power = ops.power(beta_2, local_step)
        
#        print(beta_2_power,u_product_t_1)
        
        m = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]
        
#        print(m[slicer].shape)
#        print(v[slicer].shape)
        
        nonzero_m = m[slicer]
        nonzero_v = v[slicer]
        
#        print("orig",nonzero_m.shape)
#        print("new",nonzero_m + (nonzero_gradient - nonzero_m) * (1 - beta_1))
        
        m = scatter_update_simple(m, slicer[0], nonzero_m + (nonzero_gradient - nonzero_m) * (1 - beta_1))
        v = scatter_update_simple(v, slicer[0], nonzero_v + (ops.square(nonzero_gradient) - nonzero_v) * (1 - beta_2))
        
#        m.assign(m + (gradient - m) * (1 - beta_1))
#        v.assign(v + (ops.square(gradient) - v) * (1 - beta_2))
        m_hat = u_t_1 * m / (1 - u_product_t_1) + (1 - u_t) * gradient / (
            1 - u_product_t
        )
        v_hat = v / (1 - beta_2_power)
        
        nonzero_m_hat = m_hat[slicer]
        nonzero_v_hat = v_hat[slicer]
        
#        print(slicer)
#        print(arg_slicer)
#        print(variable - (m_hat * lr) / (ops.sqrt(v_hat) + self.epsilon))
#        print((nonzero_variable - (nonzero_m_hat * lr) / (ops.sqrt(nonzero_v_hat) + self.epsilon)).shape)
#        print(nonzero_variable.shape)
#        print(variable[arg_slicer].shape)
        variable = scatter_update_simple(variable, slicer[0], nonzero_variable - (nonzero_m_hat * lr) / (ops.sqrt(nonzero_v_hat) + self.epsilon))
        
#        variable.assign(
#            variable - (m_hat * lr) / (ops.sqrt(v_hat) + self.epsilon)
#        )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
            }
        )
        return config

    
    
def scatter_update_simple(inputs, indices, updates):
    inputs = convert_to_tensor(inputs)
    indices = convert_to_tensor(indices, dtype="int64")
    updates = convert_to_tensor(updates)
    #print(indices)
    #indices = torch.transpose(indices, 0, 1)
    #print(tuple(indices))
    #print(inputs[indices].shape)
    inputs[indices] = updates
    return inputs

