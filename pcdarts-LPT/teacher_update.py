import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from teacher import *


CIFAR_CLASSES = 10
CIFAR100_CLASSES = 100


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Teacher_Updater(object):

  def __init__(self, w, h, v, args):
    self.network_momentum = args.momentum
    self.network_weight_decay_w = args.weight_decay_w
    self.network_weight_decay_h = args.weight_decay_h
    self.model_w = w
    self.model_h = h
    self.args = args
    self.model_v = v
    self.optimizer = torch.optim.Adam(self.model_v.parameters(),
                                      lr=args.model_v_learning_rate,
                                      betas=(0.5, 0.999),
                                      weight_decay=args.model_v_weight_decay)

  def _compute_unrolled_model(self,
                              criterion,
                              input, target,
                              input_external, target_external,
                              eta_w,
                              eta_h,
                              network_optimizer,
                              optimizer_w, optimizer_h):
    teacher_logits = self.model_h(self.model_w(input))
    left_loss = criterion(teacher_logits, target)

    teacher_features = self.model_w(input_external)
    teacher_logits_external = self.model_h(teacher_features)
    right_loss = F.cross_entropy(
        teacher_logits_external, target_external, reduction='none')
    binary_scores_external = self.model_v(teacher_features)
    binary_weight_external = F.softmax(binary_scores_external, 1)
    right_loss = self.args.weight_gamma * \
        binary_weight_external[:, 1] * right_loss
    loss = left_loss + right_loss.mean()

    theta_w = _concat(self.model_w.parameters()).data
    theta_h = _concat(self.model_h.parameters()).data
    try:
      moment_w = _concat(optimizer_w.state[v]['momentum_buffer']
                         for v in self.model_w.parameters()).mul_(self.network_momentum)
      moment_h = _concat(optimizer_h.state[v]['momentum_buffer']
                         for v in self.model_h.parameters()).mul_(self.network_momentum)
    except:
      moment_w = torch.zeros_like(theta_w)
      moment_h = torch.zeros_like(theta_h)

    loss.backward()
    grad_w = [v.grad.data for v in self.model_w.parameters()]
    grad_h = [v.grad.data for v in self.model_h.parameters()]
    dtheta_w = _concat(grad_w).data + self.network_weight_decay_w * theta_w
    unrolled_model_w = self._construct_model_from_theta_w(
        theta_w.sub(eta_w, moment_w + dtheta_w))

    dtheta_h = _concat(grad_h).data + self.network_weight_decay_h * theta_h
    # import ipdb; ipdb.set_trace()
    unrolled_model_h = self._construct_model_from_theta_h(
        theta_h.sub(eta_h, moment_h + dtheta_h))
    return unrolled_model_w, unrolled_model_h

  def step(self,
           criterion,
           input_train,
           target_train,
           input_valid,
           target_valid,
           input_external,
           target_external,
           eta_w,
           eta_h,
           network_optimizer,
           optimizer_w,
           optimizer_h,
           architect,
           unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(
          criterion,
          input_train, target_train,
          input_valid, target_valid,
          input_external, target_external,
          eta_w,
          eta_h,
          network_optimizer,
          optimizer_w,
          optimizer_h,
          architect)
    else:
        teacher_logits = self.model_h(self.model_w(input_valid))
        left_loss = self.args.weight_lambda * criterion(
            teacher_logits, target_valid)

        model_logits_external = architect(input_external)
        right_loss = F.cross_entropy(
            model_logits_external, target_external, reduction='none')
        binary_scores_external = self.model_v(self.model_w(input_external))
        binary_weight_external = F.softmax(binary_scores_external, 1)
        right_loss = - binary_weight_external[:, 1] * right_loss
        loss = left_loss + right_loss.mean()
        loss.backward()
    self.optimizer.step()

  def _backward_step_unrolled(self,
                              criterion,
                              input_train,
                              target_train,
                              input_valid,
                              target_valid,
                              input_external,
                              target_external,
                              eta_w,
                              eta_h,
                              network_optimizer,
                              optimizer_w,
                              optimizer_h,
                              architect):
    unrolled_model_w, unrolled_model_h = self._compute_unrolled_model(
        criterion,
        input_train, target_train,
        input_external, target_external,
        eta_w,
        eta_h,
        network_optimizer,
        optimizer_w, optimizer_h)
    output_valid = unrolled_model_h(unrolled_model_w(input_valid))
    unrolled_loss_right = criterion(output_valid, target_valid)

    unrolled_loss_right.backward()
    vector_w = [v.grad.data for v in unrolled_model_w.parameters()]
    vector_h = [v.grad.data for v in unrolled_model_h.parameters()]

    # the gradient for the second term.
    implicit_grads_w = self._hessian_vector_product_w(
        vector_w, input_external, target_external)
    implicit_grads_w = [(- self.args.weight_gamma *
                         self.args.weight_lambda * eta_w) * item for item in implicit_grads_w]

    implicit_grads_h = self._hessian_vector_product_h(
        vector_h, input_external, target_external)
    implicit_grads_h = [(- self.args.weight_gamma *
                         self.args.weight_lambda * eta_h) * item for item in implicit_grads_h]

    implicit_grads_second = [item_w + item_h for item_w,
                             item_h in zip(implicit_grads_w, implicit_grads_h)]

    # the gradient for the first term.
    student_logits_external = architect(input_external)
    student_loss = F.cross_entropy(
        student_logits_external, target_external, reduction='none')
    binary_scores_external = self.model_v(unrolled_model_w(input_external))
    binary_weight_external = F.softmax(binary_scores_external, 1)
    student_loss = binary_weight_external[:, 1] * student_loss
    student_loss = student_loss.mean()

    student_loss.backward()
    gradient_left = [- v.grad.data for v in self.model_v.parameters()]
    vector_w_prime = [v.grad.data for v in unrolled_model_w.parameters()]
    implicit_grads_first = self._hessian_vector_product_w(
        vector_w_prime, input_external, target_external)
    implicit_grads_first = [
        (self.args.weight_gamma * eta_w) * item for item in implicit_grads_first]

    implicit_grads = [
        item1 + item2 + item3 for item1, item2, item3 in zip(
            implicit_grads_first, implicit_grads_second, gradient_left)]
    # import ipdb; ipdb.set_trace()

    # update the parameters of h.
    for v, g in zip(self.model_v.parameters(), implicit_grads):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta_w(self, theta):
    if self.args.teacher_arch == '18':
      model_new = resnet18().cuda()
    elif self.args.teacher_arch == '34':
      model_new = resnet34().cuda()
    elif self.args.teacher_arch == '50':
      model_new = resnet50().cuda()
    elif self.args.teacher_arch == '101':
      model_new = resnet101().cuda()
    model_dict = self.model_w.state_dict()

    params, offset = {}, 0
    for k, v in self.model_w.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset + v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _construct_model_from_theta_h(self, theta):
    if self.args.is_cifar100:
      model_new = nn.Linear(
        512 * self.model_w.block.expansion, CIFAR100_CLASSES).cuda()
    else:
      model_new = nn.Linear(
        512 * self.model_w.block.expansion, CIFAR_CLASSES).cuda()
    model_dict = self.model_h.state_dict()

    params, offset = {}, 0
    for k, v in self.model_h.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset + v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product_w(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    # print(R)
    for p, v in zip(self.model_w.parameters(), vector):
      p.data.add_(R, v)

    teacher_features = self.model_w(input)
    teacher_logits_external = self.model_h(teacher_features)
    right_loss = F.cross_entropy(
        teacher_logits_external, target, reduction='none')
    binary_scores_external = self.model_v(teacher_features)
    binary_weight_external = F.softmax(binary_scores_external, 1)
    right_loss = binary_weight_external[:, 1] * right_loss
    right_loss = right_loss.mean()

    grads_p = torch.autograd.grad(right_loss, self.model_v.parameters())

    for p, v in zip(self.model_w.parameters(), vector):
      p.data.sub_(2 * R, v)

    teacher_features = self.model_w(input)
    teacher_logits_external = self.model_h(teacher_features)
    right_loss = F.cross_entropy(
        teacher_logits_external, target, reduction='none')
    binary_scores_external = self.model_v(teacher_features)
    binary_weight_external = F.softmax(binary_scores_external, 1)
    right_loss = binary_weight_external[:, 1] * right_loss
    right_loss = right_loss.mean()

    grads_n = torch.autograd.grad(right_loss, self.model_v.parameters())

    for p, v in zip(self.model_w.parameters(), vector):
      p.data.add_(R, v)
    # import ipdb; ipdb.set_trace()
    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

  def _hessian_vector_product_h(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    # print(R)
    for p, v in zip(self.model_h.parameters(), vector):
      p.data.add_(R, v)

    teacher_features = self.model_w(input)
    teacher_logits_external = self.model_h(teacher_features)
    right_loss = F.cross_entropy(
        teacher_logits_external, target, reduction='none')
    binary_scores_external = self.model_v(teacher_features)
    binary_weight_external = F.softmax(binary_scores_external, 1)
    right_loss = binary_weight_external[:, 1] * right_loss
    right_loss = right_loss.mean()

    grads_p = torch.autograd.grad(right_loss, self.model_v.parameters())

    for p, v in zip(self.model_h.parameters(), vector):
      p.data.sub_(2 * R, v)

    teacher_logits_external = self.model_h(teacher_features)
    right_loss = F.cross_entropy(
        teacher_logits_external, target, reduction='none')
    binary_scores_external = self.model_v(teacher_features)
    binary_weight_external = F.softmax(binary_scores_external, 1)
    right_loss = binary_weight_external[:, 1] * right_loss
    right_loss = right_loss.mean()

    grads_n = torch.autograd.grad(right_loss, self.model_v.parameters())

    for p, v in zip(self.model_h.parameters(), vector):
      p.data.add_(R, v)

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
