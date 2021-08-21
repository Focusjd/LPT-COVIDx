import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                      lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer']
                       for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(
        loss, self.model.parameters())).data + self.network_weight_decay * theta
    unrolled_model = self._construct_model_from_theta(
        theta.sub(eta, moment + dtheta))
    return unrolled_model

  def step(self,
           input_train,
           target_train,
           input_valid,
           target_valid,
           eta,
           network_optimizer,
           model_w,
           model_v,
           unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(
          input_train, target_train, input_valid,
          target_valid, eta, network_optimizer,
          model_w, model_v)
    else:
        logits_valid = self.model(input_valid)
        loss = F.cross_entropy(
            logits_valid, target_valid, reduction='none')
        binary_scores_valid = model_v(model_w(input_valid))
        binary_weight_valid = F.softmax(binary_scores_valid, 1)
        loss = binary_weight_valid[:, 1] * loss
        loss = loss.mean()
        loss.backward()
    self.optimizer.step()

  def _backward_step_unrolled(self,
                              input_train,
                              target_train,
                              input_valid,
                              target_valid,
                              eta,
                              network_optimizer,
                              model_w,
                              model_v):
    unrolled_model = self._compute_unrolled_model(
        input_train, target_train, eta, network_optimizer)

    logits_valid = unrolled_model(input_valid)
    unrolled_loss = F.cross_entropy(
        logits_valid, target_valid, reduction='none')
    binary_scores_valid = model_v(model_w(input_valid))
    binary_weight_valid = F.softmax(binary_scores_valid, 1)
    unrolled_loss = binary_weight_valid[:, 1] * unrolled_loss
    unrolled_loss = unrolled_loss.mean()
    # unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(
        vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset + v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    # print(R)
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2 * R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
