# Fibre Pore Size Measurement

<p> Pore size program based on <code>opencv</code>, <code>skimage</code> and <code>pytorch</code>. The program measures pore sizes among fibre net in pixel and returns basic statistical analysis and a pore labelled image.</p>

## Example

![result](result.jpg)

<p> An measurement example is shown above, pores and pore size are labelled on the image. It's noticeable that some pores are not labelled, this is the shortcomings of the program right now, but it's believed that the general statistical characteristics wouldn't affected. </p>

## Usage

<p> Move all files to the root path of the project, import <code>poresMeasure</code> via: </p>

    import poresmeasure

<p> Activate module via: </p>

    poresMeasure.setup()

<p> Result and viusalisation can be done via: </p>

    areas, resultImg = poresMeasure.measure(imgPath)

    poresMeasure.resultAnalyse(areas)

## License

<p> This project is released under Apache2.0 license. </p>