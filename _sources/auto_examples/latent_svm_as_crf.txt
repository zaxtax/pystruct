

.. _latent_svm_as_crf.py:


==========
Latent SVM
==========
A Latent CRF with one node is the same as a latent multiclass SVM
Using the latent variables, we can learn non-linear models. This is the
same as a simple Latent SVM model. It would obviously be more effiencent
to implement a special case for Latent SVMs so we don't have to run an
inference procedure.


**Python source code:** :download:`latent_svm_as_crf.py <latent_svm_as_crf.py>`

.. literalinclude:: latent_svm_as_crf.py
    :lines: 11-
    