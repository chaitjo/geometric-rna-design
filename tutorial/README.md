# Getting started with 3D RNA modelling as a machine learner

**Why RNA? A personal account.**
The Covid-19 pandemic was a very challenging period for me personally. 
At the same time, I got very interested in reading about and understanding some of the science behind the disease (I had last formally studied biology in 10th grade and didn't enjoy all the fact memorisation).
Around that time in 2022, I began my PhD at Cambridge and was interested in biomolecules.
I inherently knew that I wanted to motivated my work as a machine learner by biological problems that (hopelly) matter, but I didn't know much biology so I started by reading lots of popular science books. I felt that they were far more approachable, unlike textbooks.
The pandemic was coming under control largely due to mRNA vaccines and I was feeling very inspired after reading about the scientific lives of [Jennifer Doudna](https://en.wikipedia.org/wiki/The_Code_Breaker) (Nobel Prize for CRISPR) and [Venki Ramakrishnan](https://www.google.co.uk/books/edition/Gene_Machine/sRVsDwAAQBAJ?hl=en) (Nobel Prize for Ribosomes), as well as several of [Siddhartha Mukherjee](https://en.wikipedia.org/wiki/Siddhartha_Mukherjee)'s books.

One uniting theme across everything I was reading and what was happening around me was the central role played by RNA molecules.
Everybody (including myself) in deep learning was very excited about modelling proteins after AlphaFold happened, but a chance conversation with [Roger Foo](https://www.foo-lab.sg/) suggested that RNA might be something that's currently more exciting and novel to biologists. 
I can't explain precisely why, but I found something very interesting about RNA's biochemistry, the intricate structures it folds into, and the stories of the research as well as the scientists behind them. I also found the community to be very welcoming and friendly to newcomers :)

**Disclaimer.**
This document contains a currated list of resources I've been using in order to understand RNA biology and feel motivated to work on RNA design.
I'm not an expert at all, so it is very likely that I've missed something important.
None the less, I do hope these resources will be useful for someone like myself a couple years ago!

Before you begin, if you like games, do try and play the introductory parts of [Eterna](https://eternagame.org/). I can't think of a better way to start your journey in learning about RNA while having some fun!



## 🧬 RNA Biochemistry and Structural Biology

- I can highly recommend working through the [RNA 3D Structure Course](http://tinyurl.com/RNA3DStructure) by Craig L. Zirbel and Neocles Leontis at Bowling Green State University. These self-contained notes are perfect for learning topic-by-topic at your own pace, and I find myself coming back to them frequently for reference. Here are three videos to go along with the notes:
  - Anna Marie Pyle's excellent introduction to RNA structure for iBiology: [link](https://youtu.be/WCrlm18KQ48?si=mrpkgiuKg9SRu8VF)
  - A similar lecture as part of the MIT course by Barbara Imperiali: [link](https://www.youtube.com/watch?v=s1MoBTEcVYY&list=PL3xCprBkQzoneWGiypX1QOtq7lORKE-YN&index=5&t=10s&pp=gAQBiAQB)
  - Eric Westhof's talk at CASP 15 about what makes RNA structure different from proteins: [link](https://www.youtube.com/watch?v=oVaABC2oTs0). You can feel how passionate he is about all the small but important details!

- For understanding the broader biological context, I simultaneously listened through [MIT 7.016 Introductory Biology](https://www.youtube.com/playlist?list=PLUl4u3cNGP63LmSVIVzy584-ZbjbJ-Y63), which has really passionate lecturers and starts from the basics.

- [Thoughts on how to think (and talk) about RNA structure](https://www.pnas.org/doi/full/10.1073/pnas.2112677119). An approachable yet thorough paper introducing RNA structure. I keep finding myself coming back to refer to some details in this manuscript and highly recommend it.

- Lastly, if you want to work through a textbook, I'm sure there are several nice ones out there. I've been referencing [Principles of Nucleic Acid Structure](https://link.springer.com/book/10.1007/978-1-4612-5190-3) from time to time because Pietro Liò very kindly gave me his copy.

- I'm also enjoying the new [RNA Biology Coursera course](https://www.coursera.org/learn/rna-biology/) by Rhiju Das and the Das Lab at Stanford. I would perhaps start here if I were starting from scratch.



## 🎨 RNA Design (and Biomolecule Design, more broadly)

- Geometric Deep Learning for designing protein structure is very exciting at the moment! I would start by watching the latest talk I can find by David Baker ([example](https://www.youtube.com/watch?v=XI85Gh9YXS8)) and reading/watching talks about the 3 main tools for protein structure modelling and design: AlphaFold2, ProteinMPNN, and RFdiffusion.
    - [Nazim Bouatta's lectures on AlphaFold](https://cmsa.fas.harvard.edu/event/protein-folding/).
    - [ProteinMPNN by Justas Dauparas](https://www.youtube.com/watch?v=aVQQuoToTJA).
    - [RFdiffusion by Joe Watson and David Juergens](https://www.youtube.com/watch?v=wIHwHDt2NoI).
    - A new survey which does a fantastic job overviewing these methods: [link](https://arxiv.org/abs/2310.09685).
 
- Rhiju Das's excellent talk on [3D RNA Modelling and Design](https://youtu.be/2V09ne503V0?si=eqdiKTsk90oovSzB) poses a question that then captivated me: *Can we bring to bear the success of these tools from the world of proteins to RNA?*
    - The accompanying perspective article: [link](https://www.nature.com/articles/s41592-021-01132-4).
    - [RNAMake](https://www.nature.com/articles/s41565-019-0517-8) by Joseph Yesselman introduces a (non-ML) algorithm for aligning RNA motifs like lego blocks. It is particularly interesting to get a grasp of what sort of design scenarios one may be interested in.
    - I also read through Rhiju's early works introducing the structure-based design paradigm for RNA; just sort his Google Scholar by date and scroll down. The paper on [Rosetta for RNA](https://www.nature.com/articles/nmeth.1433) felt like a very important one, in particular.
 
- Ewan McRae's talk on RNA origami, another emerging non-ML paradigm for structural RNA design through assembling modular building blocks: [link](https://www.youtube.com/watch?v=nzrBUXfvwf4).

- For something a bit different but thought-provoking, Phil Holliger's talk on evolutionary approaches to designing biomolecules: [link](https://youtu.be/a4v1IbK475s?si=ud1LXCb4-1E1OpkA). "Evolution is the most powerful algorithm currently known to man", and its perhaps worth pondering how structure-based or de-novo design can augment, automate, or complement parts of the already very powerful directed evolution approach to designing biomolecules. Phil also briefly discusses [XNAs](https://en.wikipedia.org/wiki/Xeno_nucleic_acid) which are designed nucleic acids with rather profound implications for the origins of life itself; I do encourage you to go down that rabbit hole!



## 📦 Datasets

- [RNASolo](https://rnasolo.cs.put.poznan.pl/), a repository of processed PDB-derived RNA 3D structures. Go ahead and look at some of the RNA structures in the viewer, see if you like how they twist and turn!
- [RNA 3D Hub](http://rna.bgsu.edu/rna3dhub/), a repository of RNA structural annotations, motifs, and non-redundant (clustered) sets.
- Introduction to the PDB file format: [link](https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html).



## 📝 More Papers

- [Coarse-grained modeling of RNA 3D structure](https://www.sciencedirect.com/science/article/pii/S1046202316301050), an excellent overview of strategies for representing RNA structures as input for computational pipelines. Focussed on folding/structure prediction, but coarse-graining is a universal idea applicable across all machine learning tasks for RNA 3D structure.

- [The roles of structural dynamics in the cellular functions of RNAs](https://www.nature.com/articles/s41580-019-0136-0) and [RNA conformational propensities determine cellular activity](https://www.nature.com/articles/s41586-023-06080-x), two important papers advancing a growing understanding of how RNAs (and other biomolecules for that matter) are not rigid 3D objects but rather a dancing ensemble composed of multiple structural or functional states. We were inspired to build multi-state Graph Neural Networks for gRNAde based on this line of work.

- Some interesting surveys:
  - [RNA-based therapeutics: an overview and prospectus](https://www.nature.com/articles/s41419-022-05075-2).
  - [Tailor made: the art of therapeutic mRNA design](https://www.nature.com/articles/s41573-023-00827-x).

- My playlist with most of the videos references in this document: [link](https://youtube.com/playlist?list=PL3xCprBkQzoneWGiypX1QOtq7lORKE-YN&si=US_-b6pBhAWe5ziP).

