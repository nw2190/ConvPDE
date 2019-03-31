#! /usr/bin/python
# -*- coding: utf-8 -*-
"""AMSGrad Implementation based on the paper: "On the Convergence of Adam and Beyond" (ICLR 2018)
Article Link: https://openreview.net/pdf?id=ryQu7f-RZ
Original Implementation by: https://github.com/taki0112/AMSGrad-Tensorflow
"""

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer

from tensorflow import random_normal
from tensorflow import clip_by_value

    
class SGLD(optimizer.Optimizer):
    """Implementation of the AMSGrad optimization algorithm.

    See: `On the Convergence of Adam and Beyond - [Reddi et al., 2018] <https://openreview.net/pdf?id=ryQu7f-RZ>`__.

    Parameters
    ----------
    learning_rate: float
        A Tensor or a floating point value.  The learning rate.
    beta1: float
        A float value or a constant float tensor.
        The exponential decay rate for the 1st moment estimates.
    beta2: float
        A float value or a constant float tensor.
        The exponential decay rate for the 2nd moment estimates.
    epsilon: float
        A small constant for numerical stability.
        This epsilon is "epsilon hat" in the Kingma and Ba paper
        (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper.
    use_locking: bool
        If True use locks for update operations.
    name: str
        Optional name for the operations created when applying gradients.
        Defaults to "AMSGrad".
    """

    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8, use_locking=False, name="SGLD"):
        """Construct a new Adam optimizer."""
        super(SGLD, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        self._beta1_power = None
        self._beta2_power = None

    def _create_slots(self, var_list):
        """Create all slots needed by the variables.

        Args:
        var_list: A list of `Variable` objects.
        """
        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and context.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = variable_scope.variable(self._beta1, name="beta1_power", trainable=False)
                self._beta2_power = variable_scope.variable(self._beta2, name="beta2_power", trainable=False)
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "vhat", self._name)

    def _prepare(self):
        """Create all needed tensors before applying gradients.
        
        This is called with the name_scope using the "name" that
        users have chosen for the application of gradients.
        """
        self._lr_t = ops.convert_to_tensor(self._lr)
        self._beta1_t = ops.convert_to_tensor(self._beta1)
        self._beta2_t = ops.convert_to_tensor(self._beta2)
        self._epsilon_t = ops.convert_to_tensor(self._epsilon)

    def _apply_dense(self, grad, var):
        """Add ops to apply dense gradients to `var`.
        
        Args:
        grad: A `Tensor`.
        var: A `Variable` object.
        
        Returns:
        An `Operation`.
        """
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqrt = math_ops.sqrt(vhat_t)

        #var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        N = 1e6
        L = epsilon_t
        eps = 2.0*1e-6
        g_t = 1.0 / (v_sqrt + epsilon_t)
        g_t = clip_by_value(g_t, eps, 0.5*1/eps)
        noise = random_normal((), mean=0.0, stddev=eps*g_t)
        var_update = state_ops.assign_sub(var, N * lr * eps/2.0 * g_t * m_t + 0.25*noise, use_locking=self._use_locking)
        #var_update = state_ops.assign_sub(var, N * lr * eps/2.0 * g_t * m_t , use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _resource_apply_dense(self, grad, var):
        """Add ops to apply dense gradients to the variable `handle`.

        Args:
          grad: a `Tensor` representing the gradient.
          handle: a `Tensor` of dtype `resource` which points to the variable
           to be updated.

        Returns:
          An `Operation` which updates the value of the variable.
        """
        var = var.handle
        beta1_power = math_ops.cast(self._beta1_power, grad.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, grad.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, grad.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, grad.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, grad.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, grad.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m").handle
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v").handle
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat").handle
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqrt = math_ops.sqrt(vhat_t)

        var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqrt = math_ops.sqrt(vhat_t)
        var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse(self, grad, var):
        """Add ops to apply sparse gradients to `var`.

        The IndexedSlices object passed to `grad` in this function is by default
        pre-processed in `_apply_sparse_duplicate_indices` to remove duplicate
        indices (see its docstring for details). Optimizers which can tolerate or
        have correct special cases for duplicate sparse indices may override
        `_apply_sparse_duplicate_indices` instead of this function, avoiding that
        overhead.

        Args:
          grad: `IndexedSlices`, with no repeated indices.
          var: A `Variable` object.

        Returns:
          An `Operation`.
        """
        return self._apply_sparse_shared(
            grad.values,
            var,
            grad.indices,
            lambda x, i, v: state_ops.
            scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking
            )
        )

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies([resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(grad, var, indices, self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
        """Do what is needed to finish the update.

        This is called with the `name_scope` using the "name" that
        users have chosen for the application of gradients.

        Args:
          update_ops: List of `Operation` objects to update variables.  This list
            contains the values returned by the `_apply_dense()` and
            `_apply_sparse()` calls.
          name_scope: String.  Name to use for the returned operation.

        Returns:
          The operation to apply updates.
        """
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(
                    self._beta1_power * self._beta1_t, use_locking=self._use_locking
                )
                update_beta2 = self._beta2_power.assign(
                    self._beta2_power * self._beta2_t, use_locking=self._use_locking
                )
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2], name=name_scope)
