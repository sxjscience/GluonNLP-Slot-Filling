from mxnet.gluon.loss import Loss, _apply_weighting

class SoftmaxFocalLoss(Loss):
    r"""Computes the focal loss for softmax output

    If `sparse_label` is `True` (default), label should contain integer
    category indicators:

    .. math::

        \DeclareMathOperator{softmax}{softmax}

        p = \softmax({pred})

        L = -\alpha \sum_i (1 - p_{i, {label}_i})^\gamma \log p_{i,{label}_i}

    `label`'s shape should be `pred`'s shape with the `axis` dimension removed.
    i.e. for `pred` with shape (1,2,3,4) and `axis = 2`, `label`'s shape should
    be (1,2,4).

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    alpha : float, default 0.25
    gamma : float, default 2.0
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: the prediction tensor, where the `batch_axis` dimension
          ranges over batch size and `axis` dimension ranges over the number
          of classes.
        - **label**: the truth tensor. When `sparse_label` is True, `label`'s
          shape should be `pred`'s shape with the `axis` dimension removed.
          i.e. for `pred` with shape (1,2,3,4) and `axis = 2`, `label`'s shape
          should be (1,2,4) and values should be integers between 0 and 2. If
          `sparse_label` is False, `label`'s shape must be the same as `pred`
          and values should be floats in the range `[0, 1]`.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as label. For example, if label has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, axis=-1, alpha=0.25, gamma=2.0, weight=None, batch_axis=0, **kwargs):
        super(SoftmaxFocalLoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        log_pred = F.log_softmax(pred, self._axis)
        chosen_log_pred = F.pick(log_pred, label, axis=self._axis, keepdims=True)
        chosen_pred = F.exp(chosen_log_pred)
        loss = - self._alpha * (1 - chosen_pred) ** self._gamma * chosen_log_pred
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
