.. conplex-dti documentation master file, created by
   sphinx-quickstart on Wed Apr 26 15:48:05 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ConPLex: Drug Target Interaction Prediction
===========================================
* `ConPLex Home Page`_
* `ConPLex GitHub Page`_

Sequence-based prediction of drug-target interactions has the potential to accelerate drug discovery by complementing experimental screens.
Such computational prediction needs to be generalizable and scalable while remaining sensitive to subtle variations in the inputs. 
However, current computational techniques fail to simultaneously meet these goals, often sacrificing performance on one to achieve the others. 
We develop a deep learning model, ConPLex, successfully leveraging the advances in pre-trained protein language models ("PLex") and employing a novel 
protein-anchored contrastive co-embedding ("Con") to outperform state-of-the-art approaches. ConPLex achieves high accuracy, broad adaptivity to unseen data, 
and specificity against decoy compounds. It makes predictions of binding based on the distance between learned representations, enabling predictions at the scale 
of massive compound libraries and the human proteome. Experimental testing of 19 kinase-drug interaction predictions validated 12 interactions, including four 
with sub-nanomolar affinity, plus a novel strongly-binding EPHB1 inhibitor ($K_D = 1.3nM$). Furthermore, ConPLex embeddings are interpretable, which enables 
us to visualize the drug-target embedding space and use embeddings to characterize the function of human cell-surface proteins. We anticipate ConPLex will 
facilitate novel drug discovery by making highly sensitive in-silico drug screening feasible at genome scale.

If you use ConPLex, please cite "Contrastive learning in protein language space predicts interactions between drugs and protein targets" by `Rohit Singh*`_, `Sam Sledzieski*`_, `Lenore Cowen`_, and `Bonnie Berger`_.

.. _ConPLex Home Page: http://conplex.csail.mit.edu
.. _ConPLex GitHub Page: https://github.com/samsledje/ConPLex.git
.. _`Rohit Singh*`: http://people.csail.mit.edu/rsingh/
.. _`Sam Sledzieski*`: http://samsledje.github.io/
.. _`Lenore Cowen`: http://www.cs.tufts.edu/~cowen/
.. _`Bonnie Berger`: http://people.csail.mit.edu/bab/

Table of Contents
=================
.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   configuring
   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`