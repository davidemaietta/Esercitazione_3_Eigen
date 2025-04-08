#include <iostream>
#include <Eigen/Eigen>


Eigen::VectorXd PALU(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    Eigen::PartialPivLU<Eigen::MatrixXd> palu_decomp(A);
    return palu_decomp.solve(b);
}

Eigen::VectorXd QR(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr_decomp(A);
    return qr_decomp.solve(b);
}

int main() {

    // istema 1
    Eigen::Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    Eigen::Vector2d b1(-5.169911863249772e-01, 1.672384680188350e-01);

    // sistema 2
    Eigen::Matrix2d A2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
          8.320502943378437e-01, -8.324762492991313e-01;
    Eigen::Vector2d b2(-6.394645785530173e-04, 4.259549612877223e-04);

    // sistema 3
    Eigen::Matrix2d A3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
          8.320502943378437e-01,  -8.320502947645361e-01;
    Eigen::Vector2d b3(-6.4003931328043042e-10, 4.266924591433963e-10);
    
    
    // soluzione 1
    Eigen::Vector2d x1_palu = PALU(A1,b1);
    Eigen::Vector2d x1_qr = QR(A1,b1);
    
    // soluzione 2
    Eigen::Vector2d x2_palu = PALU(A2,b2);
    Eigen::Vector2d x2_qr = QR(A2,b2);
    
    // soluzione 3
    Eigen::Vector2d x3_palu = PALU(A3,b3);
    Eigen::Vector2d x3_qr = QR(A3,b3);

	// Stampa e errore
	Eigen::Vector2d x(-1.0e+00,-1.0e+00);
	std::cout << "Soluzione sistema 1 con PALU: " << x1_palu.transpose() << std::endl;	
	std::cout << "Errore relativo soluzione 1 PALU: " << (x1_palu-x).norm()/x.norm() << std::endl;
	std::cout << "Soluzione sistema 1 con QR: " << x1_qr.transpose() << std::endl;	
	std::cout << "Errore relativo soluzione 1 QR: " << (x1_qr-x).norm()/x.norm() << std::endl << std::endl;
	
	std::cout << "Soluzione sistema 2 con PALU: " << x2_palu.transpose() << std::endl;	
	std::cout << "Errore relativo soluzione 2 PALU: " << (x2_palu-x).norm()/x.norm() << std::endl;
	std::cout << "Soluzione sistema 2 con QR: " << x2_qr.transpose() << std::endl;	
	std::cout << "Errore relativo soluzione 2 QR: " << (x2_qr-x).norm()/x.norm() << std::endl << std::endl;	
	
	std::cout << "Soluzione sistema 3 con PALU: " << x3_palu.transpose() << std::endl;	
	std::cout << "Errore relativo soluzione 3 PALU: " << (x3_palu-x).norm()/x.norm() << std::endl;
	std::cout << "Soluzione sistema 3 con QR: " << x3_qr.transpose() << std::endl;	
	std::cout << "Errore relativo soluzione 3 QR: " << (x3_qr-x).norm()/x.norm() << std::endl << std::endl;
	
	// ho stampato i vettori soluzione come vettori riga per leggibilità
	
    return 0;
}