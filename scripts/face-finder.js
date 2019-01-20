const path = require('path');
const {
  cv,
  getDataFilePath,
  drawBlueRect
} = require('./utils');

// const folderName = path.basename('./');
const image = cv.imread(getDataFilePath('Lena.jpeg'));
// const cascadePath = path.resolve(folderName, './scripts/HS.xml');
const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);

// detect faces
const { objects, numDetections } = classifier.detectMultiScale(image.bgrToGray());
console.log('faceRects:', objects);
console.log('confidences:', numDetections);

if (!objects.length) {
  throw new Error('No faces detected!');
}

// draw detection
const numDetectionsTh = 10;
objects.forEach((rect, i) => {
  const thickness = numDetections[i] < numDetectionsTh ? 1 : 2;
  let nextRect = rect;
  nextRect.width = 160;
  drawBlueRect(image, nextRect, { thickness });
});

console.log(image);

cv.imwrite('new.jpg', image);
// cv.imwrite('new1.jpg', new cv.Mat(objects[0]));

const findFace = (imgPath) => {
  const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);

  const { objects, numDetections } = classifier.detectMultiScale(image.bgrToGray());
  if (!objects.length) {
    throw new Error('No faces detected!');
  }

  const numDetectionsTh = 10;
  objects.forEach((rect, i) => {
    const thickness = numDetections[i] < numDetectionsTh ? 1 : 2;
    drawBlueRect(image, rect, { thickness });
  });

  cv.imwrite('new.jpg', image);
};