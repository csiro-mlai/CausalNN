# CSIRO MLAI paper starter kit

Workflows for common data science paper writing problems.

See [https://github.com/csiro-mlai/paper-starter-kit](https://github.com/csiro-mlai/paper-starter-kit).

### LaTeX

This part was derived from [Georg Kour’s `arxiv-style`](https://github.com/kourgeorge/arxiv-style), which is a popular overleaf style.

The project hosts an aesthetic an simple LaTeX style suitable for “preprint” publications such as arXiv and bio-arXiv, etc.
It is based on NeurIPS’ [**nips_2018.sty**](https://media.nips.cc/Conferences/NeurIPS2018/Styles/nips_2018.sty) style.

This styling maintains the aesthetic of NeurIPS but adding and changing features to make it (IMO) even better and more suitable for preprints.
The result looks fairly different from NeurIPS style so that readers won't get confused to think that the preprint was published in NeurIPS.

#### Why NeurIPS?

Because the NeurIPS styling is a comfortable single column format that is convenient for reading and not sepdning your entire time tryign to fit equations into some wacky double column format.

#### Usage:

1. Use Document class **article**.
2. Copy **arxiv.sty** to the folder containing your tex file.
3. add `\usepackage{arxiv}` after `\documentclass{article}`.
4. The only packages used in the style file are **geometry** and **fancyheader**. Do not reimport them.

See **template.tex**

#### Project files:

1. **arxiv.sty** - the style file.
2. **template.tex** - a sample template that uses the **arxiv style**.
3. **references.bib** - the bibliography source file for template.tex.
4. **template.pdf** - a sample output of the template file that demonstrated the design provided by the arxiv style.


#### Handling References when submitting to arXiv.org

The most convenient way to manage references is using an external BibTeX file and pointing to it from the main file.
However, this requires running the [bibtex](http://www.bibtex.org/) tool to "compile" the `.bib` file and create `.bbl` file containing "bibitems" that can be directly inserted in the main tex file.
Arxiv [does not run bibtex](https://arxiv.org/help/submit_tex#bibtex).
You can work around this by creating a single self-contained .tex file that contains the references.
This can be done by running the BibTeX command on your machine and insert the content of the generated `.bbl` file into the `.tex` file and commenting out the `\bibliography{references}` that point to the external references file.

Below are the commands that should be run in the project folder:
1. Run `$ latex template`
2. Run `$ bibtex template`
3. A `template.bbl` file will be generated (make sure it is there)
4. Copy the `template.bbl` file content to `template.tex` into the `\begin{thebibliography}` command.
5. Comment out the `\bibliography{references}` command in `template.tex`.
6. You ready to submit to arXiv.org.


#### General Notes:
1. For help, comments, praises, bug reporting or change requests, you can contact the author at: kourgeorge/at/gmail.com.
2. You can use, redistribute and do whatever with this project, however, the author takes no responsibility on whatever usage of this project.
3. If you start another project based on this project, it would be nice to mention/link to this project.
4. You are very welcome to contribute to this project.
5. A good looking 2 column template can be found in https://github.com/brenhinkeller/preprint-template.tex.


## Advanced usage

If you cloned this as a "template repository" you can still keep up-to-date with changes from that version by tracking that as an upstream remote.

```shell
git remote add starter-kit git@github.com:csiro-mlai/starter-kit.git
git fetch starter-kit
git merge --allow-unrelated-histories starter-kit/master
git merge starter-kit master
```
