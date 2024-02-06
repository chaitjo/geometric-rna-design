# Getting started with 3D RNA modelling as a machine learner


But before you begin, if you like games, do try and play the introductory parts of [Eterna](https://eternagame.org/). I can't think of a better way to start learning about RNA while having some fun!

## 🧬 RNA Biochemistry and Structural Biology

- I can highly recommend working through the [RNA 3D Structure Course](http://tinyurl.com/RNA3DStructure) by Craig L. Zirbel and Neocles Leontis at Bowling Green State University. These self-contained notes are perfect for learning topic by topic at your own pace, and I find myself coming back to them frequently for reference. Here are two videos to go along with the notes:
  - Anna Marie Pyle's excellent introduction to RNA structure for iBiology: [link](https://youtu.be/WCrlm18KQ48?si=mrpkgiuKg9SRu8VF)
  - A similar lecture as part of the MIT course by Barbara Imperiali: [link](https://www.youtube.com/watch?v=s1MoBTEcVYY&list=PL3xCprBkQzoneWGiypX1QOtq7lORKE-YN&index=5&t=10s&pp=gAQBiAQB)

- For understanding the broader biological context, I simultaneously listened through [MIT 7.016 Introductory Biology](https://www.youtube.com/playlist?list=PLUl4u3cNGP63LmSVIVzy584-ZbjbJ-Y63), which has really passionate lecturers and starts from the basics.

- Lastly, if you want to work through a textbook, I'm sure there are several nice ones out there. I've been referencing [Principles of Nucleic Acid Structure](https://link.springer.com/book/10.1007/978-1-4612-5190-3) from time to time because Pietro Liò very kindly gave me his copy.

- I'm also enjoying the new [RNA Biology Coursera course](https://www.coursera.org/learn/rna-biology/) by Rhiju Das and the Das Lab at Stanford.

---

## 🎨 RNA Design (and Biomolecule Design, more broadly)

- I would start by watching the latest talk I can find by David Baker ([example](https://www.youtube.com/watch?v=XI85Gh9YXS8)) and reading/watching talks about the 3 main tools for protein structure modelling and design: AlphaFold2, ProteinMPNN, and RFdiffusion.
    - [Nazim Bouatta's lectures on AlphaFold](https://cmsa.fas.harvard.edu/event/protein-folding/) 
    - [ProteinMPNN by Justas Dauparas](https://www.youtube.com/watch?v=aVQQuoToTJA)
    - [RFdiffusion by Joe Watson and David Juergens](https://www.youtube.com/watch?v=wIHwHDt2NoI)
    - A new survey which does a fantastic job overviewing these methods: [link](https://arxiv.org/abs/2310.09685)
 
- Rhiju Das's excellent talk on [3D RNA Modelling and Design](https://youtu.be/2V09ne503V0?si=eqdiKTsk90oovSzB) poses a question that then captivated me: Can we bring to bear the success of these tools from the world of proteins to RNA?

- Phil Holliger's talk on evolutionary approaches to designing biomolecules: [link](https://youtu.be/a4v1IbK475s?si=ud1LXCb4-1E1OpkA). "Evolution is the most powerful algorithm currently known to man", and its perhaps worth pondering how structure-based or de-novo design can augment, automate, or complement parts of the already very powerful directed evolution approach to designing biomolecules.

---

## 📦 Datasets

- Introduction to the PDB file format: [link](https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html).
- [RNASolo](https://rnasolo.cs.put.poznan.pl/), a repository of processed PDB-derived RNA 3D structures.
- [RNA 3D Hub](http://rna.bgsu.edu/rna3dhub/), a repository of RNA structural annotations, motifs, and non-redundant (clustered) sets.

---

## 📝 More Papers

- [Coarse-grained modeling of RNA 3D structure](https://www.sciencedirect.com/science/article/pii/S1046202316301050), an excellent overview of strategies for representing RNA structures as input for computational pipelines. Focussed on folding/structure prediction, but coarse-graining is a universal idea applicable across all machine learning tasks for RNA 3D structure.

---

## 🎥 More Videos

- Eric Westof's talk at CASP 15, about what makes RNA structure different from proteins: [link](https://www.youtube.com/watch?v=oVaABC2oTs0) 

