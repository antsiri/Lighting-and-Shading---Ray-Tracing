# Tau v2.3.1 (10/04/2024)

This LaTeX class, named 'tau', is designed to provide a clean and professional layout for academic articles and lab reports. It offers clarity in code structure, making it easy to understand and customize according to your specific needs.

## License

Creative Commons CC BY 4.0

## Features

* The 'tau.cls' file can be modified to your preferences and requirements.
* Stix2 font for clear text.
* Includes custom environments for notes, information, and code.
* ToC which provides a hierarchical structure for the document contents.
* Fancyfoot/head that includes contextual information such as title, institution, date, etc.
* The listings package to insert code with syntax highlighting and line numbering for various languages (Matlab, C, C++, and LaTeX).
* Appropriate conjunction ('y' for Spanish, 'and' for English) when two authors are included.

## Bibliography in external editors

If the bibliography does not show up the first time that you compile the document, try running the 'tau.cls' and 'tau.bib' file with biber from the MikTeX console or your preferred LaTeX distribution to generate the auxiliar files and (re)run the main.tex.

## Updates Log

**Version 1.0 (01/03/2024)**

Launch of the first edition of tau-book class, made especially for academic articles and laboratory reports. 

**Version 2.0 (03/03/2024)** 

[1] The table of contents has a new design.
[2] Figure, table and code captions have a new style.

**Version 2.1 (04/03/2024)** 

[1] All URLs have the same font format.
[2] Corrections to the author "and" were made.
[3] Package name changed from kappa.sty to tau.sty.

**Version 2.2 (15/03/2024)** 

[1] Tau-book is dressed in midnight blue for title, sections, URLs, and more.
[2] The \abscontent{} command was removed and the abstract was modified directly.
[3] The title is now centered and lines have been removed for cleaner formatting.
[4] New colors and formatting when inserting code for better appearance.

**Version 2.3 (08/04/2024)** 

[1] Class name changed from tau-book to just tau. 
[2] A new code for the abstract was created.
[3] The abstract font was changed from italics to normal text keeping the keywords in italics.
[4] Taublue color was changed.
[5] San Serif font was added for title, abstract, sections, captions and environments.
[6] The table of contents was redesigned for better visualization.
[7] The new environment (tauenv) with customized title was included.
[8] The appearance of the header was modified showing the title on the right or left depending on the page.
[9] New packages were added to help Tikz perform better.
[10] The pbalance package was added to balace the last two columns of the document (optional).
[11] The style of the fancyfoot was changed by eliminating the lines that separated the information.
[12] New code was added to define the space above and below in equations. 

**Version 2.3.1 (10/04/2024)** 

[1] We say goodbye to tau.sty.
[2] Introducing tauenvs package which includes the defined environments.
[3] The packages that were in tau.sty were moved to the class document (tau.cls).

## Contact

If you have any comments to improve tau class in future versions or found a bug, do not doubt to contact me.

*Email: memo.notess1@gmail.com*
*Instagram: memo.notess*
*Site: https://sites.google.com/view/memo-notess/p%C3%A1gina-principal*

-------
Enjoy writing with tau class :D