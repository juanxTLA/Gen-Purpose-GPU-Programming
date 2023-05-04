#ifndef __Matrix_hpp
#define __Matrix_hpp


// this is mainly just a handle for memory and will
// allow us to easily index into memory and query information
// about sizes and such
//
// should be utilized with shared pointers instead
template <typename Datatype>
class Matrix {
  public:
    
    Matrix(
      int numberOfRows,
      int numberOfCols
    ) :
        numberOfRows_(numberOfRows),
        numberOfCols_(numberOfCols),
        data_(new Datatype[numberOfRows * numberOfCols])
    {}

    void print() {
      for (int rowIdx = 0; rowIdx < numberOfRows_; ++rowIdx) {
        for (int colIdx = 0; colIdx < numberOfCols_; ++colIdx) {
          printf("%6.3f\t", (double)((*this)(rowIdx, colIdx)));
        }
        printf("\n");
      }
    }

    int numberOfRows() {
      return numberOfRows_;
    }
    
    int numberOfCols() {
      return numberOfCols_;
    }

    Datatype & operator() (
      int rowIdx,
      int colIdx
    ) {
      return data_[rowIdx * numberOfCols_ + colIdx];
    }
  
    int leadingDimension() {
      return numberOfCols_;
    }

    Datatype * data() { 
      return data_;
    }

    // we have to clean up the memory we allocated 
    ~Matrix() {
      delete[] data_;
    }
    
    // so we don't ever allow ourselves to do copies,
    // we will only interact with this class via shared
    // pointers
    Matrix & operator = (Matrix const &) = delete;
    Matrix & operator = (Matrix &&) = delete;
    Matrix(Matrix const &) = delete;
    Matrix(Matrix &&) = delete;

  private:
    int numberOfRows_;
    int numberOfCols_;
    Datatype * data_;
};

#endif
